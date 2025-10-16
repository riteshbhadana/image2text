import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import requests
import os

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Cache the models to avoid reloading
@st.cache_resource
def load_img2text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    """Load GPT-2 for local story generation"""
    return pipeline(
        "text-generation", 
        model="gpt2",
        pad_token_id=50256
    )

def img2text(image_path):
    """Convert image to text description"""
    try:
        image_to_text = load_img2text_model()
        text = image_to_text(image_path)[0]["generated_text"]
        return text
    except Exception as e:
        st.error(f"Error in image-to-text: {e}")
        return None

def generate_story(scenario):
    """Generate a creative 20-word story DIRECTLY related to image description"""
    if not scenario:
        return None
    
    try:
        story_generator = load_story_model()
        
        # Simpler, more reliable prompt
        prompt = f"{scenario}. The story begins:"
        
        result = story_generator(
            prompt,
            max_length=80,  # Reduced for shorter output
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=50256,
            repetition_penalty=1.5,
            no_repeat_ngram_size=2
        )
        
        # Extract story
        full_text = result[0]['generated_text']
        story = full_text.replace(prompt, "").strip()
        
        # Clean the story
        story = story.split('\n')[0]
        story = story.replace('  ', ' ')
        story = story.strip()
        
        # If story is empty or too short, create a simple one
        if len(story) < 10:
            # Fallback: create a simple descriptive story
            story = create_simple_story(scenario)
        else:
            # Get 18-20 words
            words = story.split()[:20]
            story = " ".join(words)
        
        # Ensure proper ending
        if story and not story.endswith(('.', '!', '?')):
            story += '.'
        
        return story if story else create_simple_story(scenario)
    
    except Exception as e:
        st.warning(f"Model error: {str(e)[:100]}")
        # Use fallback
        return create_simple_story(scenario)

def create_simple_story(scenario):
    """Create a simple template-based story when model fails (18-20 words)"""
    try:
        # Simple story templates - all around 18-20 words
        templates = [
            "This moment captured pure joy. The scene unfolded naturally, revealing beauty in everyday life and creating lasting memories.",
            "Something magical happened here. Every detail told its own story. The atmosphere felt warm and inviting throughout.",
            "In this peaceful moment, everything aligned perfectly. Nature and comfort blended together, creating harmony and contentment.",
            "Life's beautiful moments appear unexpectedly. This scene embodied warmth. Each element contributed to the peaceful atmosphere.",
            "A snapshot of happiness captured here. The setting provided comfort and joy, making everything come alive beautifully."
        ]
        
        import random
        story = random.choice(templates)
        
        return story
    except:
        return "A beautiful moment captured in time. Everything seemed perfect in this peaceful scene worth remembering."

def text2speech_gtts(message, output_file="audio.mp3"):
    """Convert text to speech using gTTS - optimized for ~10 seconds"""
    if not message:
        return False
    
    try:
        from gtts import gTTS
        
        # Create speech with slightly slower speed for clarity
        tts = gTTS(text=message, lang='en', slow=False)
        tts.save(output_file)
        return True
    
    except ImportError:
        st.error("gTTS not installed. Run: pip install gtts")
        return False
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        return False

def text2speech_pyttsx3(message, output_file="audio.mp3"):
    
    if not message:
        return False
    
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Optional: adjust voice properties
        engine.setProperty('rate', 150)    # Speed
        engine.setProperty('volume', 0.9)  # Volume
        
        # Save to file
        engine.save_to_file(message, output_file)
        engine.runAndWait()
        return True
    
    except ImportError:
        st.error("pyttsx3 not installed. Run: pip install pyttsx3")
        return False
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        return False

def main():
    st.set_page_config(page_title="Image to Audio Story", page_icon="🤖")
    
    st.header("Turn Image into Audio Story 🎭🔊")
    st.write("Upload an image and I'll create a story and narrate it for you!")
    
    # Info box
    st.info("✨ **100% FREE** -! Stories are 18-20 words")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        # Save the uploaded file temporarily
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        
        # Process button
        if st.button("🚀 Generate Story and Audio", type="primary"):
            with st.spinner("🔍 Analyzing image..."):
                scenario = img2text(uploaded_file.name)
            
            if scenario:
                # Show image description first
                with st.expander("📝 What AI Sees in Your Image", expanded=True):
                    st.info(f"🔍 **{scenario}**")
                    st.caption("↓ The story continues from this scene ↓")
                
                with st.spinner("✍️ Creating story from the image scene..."):
                    story = generate_story(scenario)
                
                if story:
                    st.success("✅ Story generated!")
                    
                    # Count words and check uniqueness
                    words = story.split()
                    word_count = len(words)
                    unique_words = len(set(words))
                    
                    with st.expander("📖 Generated Story (Continues from Image)", expanded=True):
                        # Show the full narrative
                        st.write("**Complete Story:**")
                        st.write(f"*{scenario}. {story}*")
                        
                        st.divider()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"📊 Total words: {word_count}")
                        with col2:
                            st.caption(f"✨ Unique words: {unique_words}")
                    
                    with st.spinner("🎙️ Creating audio narration..."):
                        # Narrate the complete story with context
                        full_narrative = f"{scenario}. {story}"
                        success = text2speech_gtts(full_narrative, "audio.mp3")
                        audio_file = "audio.mp3"
                    
                    if success and os.path.exists(audio_file):
                        st.success("✅ Audio generated successfully!")
                        st.audio(audio_file)
                        
                        # Download button
                        with open(audio_file, "rb") as audio:
                            st.download_button(
                                label="📥 Download Audio",
                                data=audio,
                                file_name="story_audio.mp3",
                                mime="audio/mp3"
                            )
                    else:
                        st.error("Failed to generate audio. Make sure gtts is installed: pip install gtts")
                        
                        # Show debug info
                        with st.expander("🔧 Troubleshooting"):
                            st.write("If audio fails:")
                            st.code("pip install --upgrade gtts")
                            st.write("Make sure you have internet connection for gTTS.")
                else:
                    st.error("⚠️ Story generation had issues, but don't worry!")
                    st.info("This can happen with the free GPT-2 model. Try:")
                    st.write("1. Click the button again")
                    st.write("2. Try a different image")
                    st.write("3. Restart the app if problem persists")
            else:
                st.error("Failed to analyze image. Please try another image.")
            
            # Clean up temporary file
            try:
                os.remove(uploaded_file.name)
            except:
                pass
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        **How it works:**
        1. 📸 Upload an image
        2. 🤖 AI describes what it sees (BLIP)
        3. ✍️ AI writes a 18-20 word story BASED ON the image
        4. 🔊 AI narrates the story (~8 seconds)
        
        """)
        
        
        
        st.divider()
        
        st.subheader("💡 Tips")
        st.write("• Upload clear images for better descriptions")
        st.write("• AI describes the image first")
        st.write("• Story is created based on what AI sees")
        st.write("• Audio narrates the story (~8 seconds)")
        st.write("• Click 'Download Audio' to save")

if __name__ == '__main__':
    main()