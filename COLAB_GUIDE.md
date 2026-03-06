# Running MangaMotion on Google Colab (T4 GPU)

## Step 1: Create a New Colab Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a **New Notebook**
3. Go to **Runtime → Change runtime type → T4 GPU** → Save

---

## Step 2: Clone Your Repo

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME
```

> Replace `YOUR_USERNAME/YOUR_REPO_NAME` with your actual GitHub repo path.

---

## Step 3: Install Python Dependencies

```python
!pip install -r requirements.txt
```

---

## Step 4: Install & Start Ollama

```python
# Install Ollama
!curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server in background (GPU will be auto-detected)
import subprocess
subprocess.Popen(["ollama", "serve"])

# Wait for server to start
import time
time.sleep(5)

# Pull the AI models (~6GB total, takes 2-3 min)
!ollama pull moondream
!ollama pull mistral:7b
```

---

## Step 5: Verify Ollama is Running

```python
!curl http://localhost:11434/api/tags
```

You should see a JSON response listing the two models.

---

## Step 6: Launch the App

```python
!python app.py
```

Gradio will print a **public URL** (something like `https://xxxxx.gradio.live`).  
Click it to open the UI in your browser.

---

## Step 7: Use the App

1. Upload a manga PDF
2. Pick reading direction (RTL for Japanese manga)
3. Choose TTS engine (`edge` recommended — works on Colab)
4. Click **Generate Video**
5. Download the output MP4

---

## Tips

- **Session resets**: When Colab disconnects, you lose everything. Re-run Steps 2–6.
- **Disk space**: Free Colab has ~78GB. Models use ~6GB, so plenty of room.
- **Speed**: T4 GPU makes Ollama analysis ~5-10x faster than CPU.
- **TTS**: Use `edge` (best quality, needs internet) or `pyttsx3` (may need `!apt install espeak` first).
- **If pyttsx3 fails**: Run `!apt install espeak libespeak-dev` before Step 6.
