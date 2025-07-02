import validators
import streamlit as st
import yt_dlp
import re
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Summarize from YT video or Website", page_icon=":video_camera:", layout="wide")

def extract_youtube_transcript(url):
    """Extract transcript from YouTube video using yt-dlp"""
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Try to get manual subtitles first, then automatic
            subtitles = info.get('subtitles', {})
            auto_subtitles = info.get('automatic_captions', {})
            
            transcript_text = ""
            
            # Try English subtitles first
            for lang in ['en', 'en-US', 'en-GB']:
                if lang in subtitles:
                    # Get the subtitle URL and extract text
                    sub_info = subtitles[lang][0]  # Get first subtitle format
                    if 'url' in sub_info:
                        import requests
                        response = requests.get(sub_info['url'])
                        if response.status_code == 200:
                            # Parse VTT or SRT content
                            content = response.text
                            # Remove VTT/SRT formatting and extract text
                            lines = content.split('\n')
                            for line in lines:
                                # Skip timestamp lines and empty lines
                                if '-->' not in line and line.strip() and not line.strip().isdigit():
                                    # Remove HTML tags if present
                                    clean_line = re.sub(r'<[^>]+>', '', line.strip())
                                    if clean_line:
                                        transcript_text += clean_line + " "
                            break
            
            # If no manual subtitles, try automatic captions
            if not transcript_text:
                for lang in ['en', 'en-US', 'en-GB']:
                    if lang in auto_subtitles:
                        sub_info = auto_subtitles[lang][0]
                        if 'url' in sub_info:
                            import requests
                            response = requests.get(sub_info['url'])
                            if response.status_code == 200:
                                content = response.text
                                lines = content.split('\n')
                                for line in lines:
                                    if '-->' not in line and line.strip() and not line.strip().isdigit():
                                        clean_line = re.sub(r'<[^>]+>', '', line.strip())
                                        if clean_line:
                                            transcript_text += clean_line + " "
                                break
            
            if transcript_text:
                # Create a Document object similar to LangChain loaders
                metadata = {
                    'title': info.get('title', 'Unknown'),
                    'uploader': info.get('uploader', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'source': url
                }
                return [Document(page_content=transcript_text.strip(), metadata=metadata)]
            else:
                return None
                
    except Exception as e:
        st.error(f"Error extracting YouTube content: {str(e)}")
        return None

st.title("Summarize from YT video or Website")
st.subheader("Enter a YouTube video URL or a website URL to summarize its content.")

with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API Key", type="password", value="")

url = st.text_input("Enter a YouTube video URL or a website URL", label_visibility="collapsed")

if st.button("Summarize"):
    if not url:
        st.error("Please enter a URL.")
    elif not validators.url(url):
        st.error("Please enter a valid URL.")
    elif not groq_api_key or groq_api_key.strip() == "":
        st.error("Please enter your Groq API Key.")
    else:
        try:
            with st.spinner("Loading content..."):
                if "youtube.com/watch" in url or "youtu.be" in url:
                    documents = extract_youtube_transcript(url)
                    if not documents:
                        st.error("Could not extract transcript from YouTube video.")
                        st.info("This may happen if:")
                        st.info("• Video has no captions/subtitles available")
                        st.info("• Video is private, age-restricted, or deleted")
                        st.info("• Regional restrictions apply")
                        st.stop()
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
                    )
                    documents = loader.load()
                
                if not documents:
                    st.error("No content found at the provided URL.")
                    st.stop()
                
                # --- CHUNKING LARGE DOCUMENTS ---
                total_length = sum(len(doc.page_content) for doc in documents)
                max_chunk_size = 4000  # characters, adjust as needed
                if total_length > 8000:
                    st.info(f"Splitting large content ({total_length:,} characters) into smaller chunks...")
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=max_chunk_size,
                        chunk_overlap=200
                    )
                    split_docs = []
                    for doc in documents:
                        split_docs.extend([
                            Document(page_content=chunk, metadata=doc.metadata)
                            for chunk in text_splitter.split_text(doc.page_content)
                        ])
                    documents = split_docs
                # --- END CHUNKING ---

                st.success("Content loaded successfully!")
                
                llm = ChatGroq(
                    model="llama3-8b-8192",
                    api_key=groq_api_key,
                    temperature=0.1,
                    max_tokens=2000,
                )
                
                prompt_template = PromptTemplate(
                    input_variables=["text"],
                    template="""
                    Please provide a comprehensive summary of the following content:
                    
                    {text}
                    
                    Summary should include:
                    - Main topics discussed
                    - Key points and important details
                    - Any conclusions or takeaways
                    
                    Please make the summary clear, concise, and well-structured.
                    """
                )
                
                # Choose chain type based on content length
                total_length = sum(len(doc.page_content) for doc in documents)
                
                if total_length > 8000:  # For longer content
                    chain_type = "map_reduce"
                    st.info(f"Processing long content ({total_length:,} characters) using map-reduce approach...")
                    
                    # Create a separate prompt for the final combination step
                    combine_prompt = PromptTemplate(
                        input_variables=["text"],
                        template="""
                        You are given multiple summaries of different parts of a video/document. 
                        Please combine these summaries into one comprehensive summary.
                        
                        Summaries to combine:
                        {text}
                        
                        Please create a unified summary that:
                        - Captures all main topics and themes
                        - Maintains logical flow and structure
                        - Eliminates redundancy while preserving important details
                        - Provides clear conclusions and takeaways
                        """
                    )
                    
                    summarize_chain = load_summarize_chain(
                        llm=llm,
                        chain_type=chain_type,
                        map_prompt=prompt_template,
                        combine_prompt=combine_prompt,
                        verbose=True
                    )
                else:  # For shorter content
                    chain_type = "stuff"
                    st.info(f"Processing content ({total_length:,} characters) using direct approach...")
                    
                    summarize_chain = load_summarize_chain(
                        llm=llm,
                        chain_type=chain_type,
                        prompt=prompt_template
                    )
                
                with st.spinner("Generating summary..."):
                    summary = summarize_chain.run(documents)
                
                st.subheader("Summary")
                st.write(summary)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your API key and URL, then try again.")

with st.sidebar:
    st.markdown("### Instructions")
    st.markdown("""
    1. Enter your Groq API Key
    2. Paste a YouTube video URL or website URL
    3. Click 'Summarize' to generate a summary
    
    **Supported URLs:**
    - YouTube videos (youtube.com/watch or youtu.be)
    - Most websites with readable content
    
    **YouTube Requirements:**
    - Video must have captions/transcripts enabled
    - Video should not be age-restricted or private
    - Some videos may be blocked due to regional restrictions
    """)
    
    st.markdown("### About")
    st.markdown("""
    This app uses:
    - **Groq**: For fast AI inference
    - **LangChain**: For document processing
    - **Streamlit**: For the web interface
    - **yt-dlp**: For YouTube transcript extraction
    """)