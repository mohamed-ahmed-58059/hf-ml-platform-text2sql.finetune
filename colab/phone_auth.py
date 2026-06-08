"""Headless / phone OAuth for the Colab CLI (no browser on this machine).

Run with the colab-cli's OWN interpreter so google_auth_oauthlib and the
bundled oauth_config.json are importable:

  PY=$(ls /root/.local/share/uv/tools/google-colab-cli/bin/python)

  # 1) get a sign-in link (open on phone, approve; the localhost page will fail
  #    to load -- that's expected -- copy the `code` value from the address bar)
  OAUTHLIB_RELAX_TOKEN_SCOPE=1 $PY colab/phone_auth.py url

  # 2) exchange the code -> writes ~/.config/colab-cli/token.json, after which
  #    `colab` refreshes headlessly with no browser
  OAUTHLIB_RELAX_TOKEN_SCOPE=1 $PY colab/phone_auth.py exchange '<code>'
"""
import json
import os
import sys

from google_auth_oauthlib.flow import Flow
import colab_cli

PKG = os.path.dirname(colab_cli.__file__)
CLIENT = json.load(open(os.path.join(PKG, "oauth_config.json")))
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
STATE = "/tmp/colab_oauth_state.json"
TOKEN = os.path.expanduser("~/.config/colab-cli/token.json")


def cmd_url():
    flow = Flow.from_client_config(CLIENT, scopes=SCOPES, redirect_uri="http://localhost")
    url, state = flow.authorization_url(access_type="offline", prompt="consent")
    json.dump({"code_verifier": flow.code_verifier, "state": state}, open(STATE, "w"))
    print(url)


def cmd_exchange(code):
    st = json.load(open(STATE))
    flow = Flow.from_client_config(CLIENT, scopes=SCOPES, redirect_uri="http://localhost",
                                   state=st["state"])
    flow.code_verifier = st["code_verifier"]
    flow.fetch_token(code=code)
    os.makedirs(os.path.dirname(TOKEN), exist_ok=True)
    open(TOKEN, "w").write(flow.credentials.to_json())
    print("token ->", TOKEN, "| refresh_token:", bool(flow.credentials.refresh_token))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "url":
        cmd_url()
    elif len(sys.argv) == 3 and sys.argv[1] == "exchange":
        cmd_exchange(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
