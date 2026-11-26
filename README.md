# DeepGuard (image screening demo)

Upload a face photo. An EfficientNet-B0 classifier gives a score and a risk tier. Optionally, a Groq-hosted
vision model adds visual commentary, which never changes the result. This assists human review; it does not
determine whether an image is authentic.

## Run on Windows (PowerShell, inside this folder). Needs Python 3.10 to 3.12.
    py -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    copy .env.example .env        # then put your Groq key in .env
    .venv\Scripts\python app.py   # open http://localhost:8000

Model file: save your checkpoint as `checkpoints\deepguard_image_best.pth`.
If Windows shows it as `deepguard_image_best.zip`, RENAME it to .pth. Do not unzip it.
The default Windows torch install runs on the CPU, which is enough for one image at a time.

## Public link (keep both windows open)
    winget install --id Cloudflare.cloudflared      # once, then open a new window
    cloudflared tunnel --url http://localhost:8000
Share the https://....trycloudflare.com address it prints. It changes each time you restart the tunnel.
Anyone with the link can use the app and your Groq quota, so close the tunnel after the demo.
Plug in the laptop and turn off sleep.

## Known limitations
- Risk tiers use hand-chosen cut-offs (0.30 / 0.70). The score is not a calibrated probability.
- Train/validation split is image-level; related frames may appear in both.
- Only images with a face found by a Haar cascade are assessed. Training also included centre crops
  where no face was found.
- The attention map shows what influenced the model, not where an image was edited.
- Visual commentary is generated text. It can be wrong and may disagree with the score.
- No video support, provenance (C2PA) check, adversarial testing or accounts.
- No uploads are saved by this app. With commentary switched on, the image is sent to Groq.

## Model evaluation
TODO: fill in from the finished source-held-out run (training log and Cell 5 output).
