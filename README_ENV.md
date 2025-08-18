Local .env and direnv setup instructions

1) Create a .env file from the example (do not commit .env):

   cp .env.example .env

   # Edit .env and paste your real API key
   # Example (zsh):
   # export OPENAI_API_KEY="sk-..."

2) Add .env to .gitignore if not already ignored:

   echo ".env" >> .gitignore

3) Use python-dotenv to load .env in development (optional):

   pip install python-dotenv

   # In Python code
   from dotenv import load_dotenv
   load_dotenv()  # loads variables from .env into os.environ

4) Use direnv for safer environment management (recommended):

   # Install direnv (Homebrew)
   brew install direnv

   # Hook direnv into your shell (zsh): add to ~/.zshrc
   echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
   source ~/.zshrc

   # Create an .envrc file in the repo root and allow it:
   # Example .envrc (do not commit):
   # export OPENAI_API_KEY="sk-..."

   # To create and allow:
   cp .env.example .envrc
   # edit .envrc to set the real key (or simply: echo 'export OPENAI_API_KEY="sk-..."' > .envrc)
   direnv allow

5) Use macOS Keychain (optional) for GUI persistence:

   # Store:
   security add-generic-password -a "$USER" -s "OPENAI_API_KEY" -w "sk-..."

   # Retrieve in scripts:
   security find-generic-password -s "OPENAI_API_KEY" -w

6) CI / Production:

   # Inject secrets via your CI provider's secret manager (GitHub Actions, GitLab CI, etc.)

Security notes:
- Never commit real API keys to the repository. If you accidentally do, rotate the key immediately.
- Prefer direnv or OS keychain over a plain .env file if possible.
