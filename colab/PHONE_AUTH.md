# Google / Colab login from your phone (headless OAuth)

Copy this into a session prompt when you need to (re)authenticate the `colab`
CLI. The CLI normally needs a desktop browser to click "Allow";
`colab/phone_auth.py` works around that with a manual PKCE flow you complete on
your phone. Two commands, one round-trip.

> The token at `~/.config/colab-cli/token.json` is **wiped on every container
> restart**, so redo this each fresh session. It's quick: url → tap → exchange.

## Steps

1. **Use the CLI's own Python** so `google_auth_oauthlib` and the bundled
   `oauth_config.json` (client credentials) are importable:

   ```bash
   PY=$(ls /root/.local/share/uv/tools/google-colab-cli/bin/python)
   ```

2. **Generate the sign-in URL.** Builds the OAuth flow (scopes: openid, profile,
   email, cloud-platform, colaboratory, **drive.file**), saves the PKCE
   `code_verifier` + `state` to `/tmp/colab_oauth_state.json`, and prints a
   Google URL:

   ```bash
   OAUTHLIB_RELAX_TOKEN_SCOPE=1 $PY colab/phone_auth.py url
   ```

3. **Open that URL on your phone**, sign in, and approve **all** boxes
   (including Drive). The redirect lands on `http://localhost`, which
   **fails to load — that's expected**; nothing is listening. The piece you need
   is the **`code=...` value in the address bar** — copy it.

4. **Exchange the code** for a token. Reads back the saved `state` /
   `code_verifier`, fetches the token, and writes
   `~/.config/colab-cli/token.json` (with a refresh token):

   ```bash
   OAUTHLIB_RELAX_TOKEN_SCOPE=1 $PY colab/phone_auth.py exchange '<code>'
   ```

After this, `colab` refreshes the token **headlessly** — no more browser.

## Gotchas

- **`OAUTHLIB_RELAX_TOKEN_SCOPE=1` is required.** Google returns a slightly
  different scope set than requested, which otherwise makes oauthlib error out
  on the exchange.
- **The `localhost` page failing is normal** — the authorization `code` is in
  the URL query string, which is all the PKCE exchange needs.
- **Token is ephemeral.** Wiped on container restart; the
  `/tmp/colab_oauth_state.json` handoff file is only needed between step 2 and
  step 4.
