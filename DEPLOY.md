# Deploying dialogen-repo online

Three free options, in order of ease. All three build straight from this repo's `requirements.txt` and `app.py`, nothing extra to configure.

## Option A: Hugging Face Spaces (recommended)

1. Create a free account at huggingface.co, then click New Space (avatar menu, top right).
2. Pick a name, set SDK to Streamlit, choose visibility (private is fine to start), click Create Space. Hugging Face gives you an empty git repo at huggingface.co/spaces/you/space-name.
3. From this project's folder:

```bash
git init
git remote add space https://huggingface.co/spaces/<you>/<space-name>
git add .
git commit -m "Initial commit"
git push space main
```

Make sure `checkpoints/hybrid_dialogue.weights.h5` and `checkpoints/tokenizer.json` are committed. They are the trained model. If they are large, run `git lfs track "*.h5"` first.

4. The Space builds automatically and gives you a public URL at huggingface.co/spaces/you/space-name. First build takes a few minutes while it installs TensorFlow.

## Option B: Streamlit Community Cloud

1. Push this repo to GitHub. Create a repo at github.com/new, then `git remote add origin <url> && git push -u origin main`.
2. Go to share.streamlit.io, sign in with GitHub, click New app.
3. Point it at your repo, branch main, main file app.py, click Deploy.
4. Free tier sleeps after inactivity but wakes on the next visit. Fine for a demo, not for something that needs to stay up.

## Option C: Render.com (Docker)

1. Push this repo to GitHub, same as Option B.
2. On render.com, New, Web Service, connect the repo, set environment to Docker. It finds the included Dockerfile automatically.
3. Deploy. Render builds the image and exposes port 8501. Free tier spins down on inactivity and cold-starts on the next request.

## Notes that apply to all three

Train before you deploy, or the app just shows the "no checkpoint found" message. The repo ships with a working demo checkpoint already in `checkpoints/`, so this only matters if you retrain.

Checkpoint size: the bundled demo checkpoint is a few MB. A version trained on the full Cornell corpus with a larger vocab will be tens of MB. All three platforms handle that, but use Git LFS for anything over about 50 MB: `git lfs track "checkpoints/*.h5"`.

None of these platforms are reachable from a sandboxed dev environment (no outbound network to huggingface.co, github.com push endpoints, or render.com), so the account creation and git push steps above have to run from your own machine.
