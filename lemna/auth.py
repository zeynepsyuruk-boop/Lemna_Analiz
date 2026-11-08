"""Password login for the web interface.

Users are read from the Streamlit secrets ([users] table in .streamlit/secrets.toml) or from the
environment variable LEMNA_USERS ("name:hash,name:hash"). Without any users the interface opens
without a login, as on a personal computer.

Create a password hash with:  python -m lemna.auth
"""

import getpass
import hashlib
import hmac
import os
import secrets
import time

ITERATIONS = 310_000
MAX_FAILS = 5        # failed attempts before a pause
LOCK_SECONDS = 60


def hash_password(password, iterations=ITERATIONS):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def verify_password(password, stored):
    try:
        method, iterations, salt, digest = stored.split("$")
        if method != "pbkdf2_sha256":
            return False
        test = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations)).hex()
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(test, digest)


def load_users(st_secrets=None):
    """Return {username: hash} from the Streamlit secrets or the environment."""
    users = {}
    try:
        if st_secrets is not None and "users" in st_secrets:
            users = {str(k).strip().lower(): str(v) for k, v in st_secrets["users"].items()}
    except Exception:  # no secrets file
        users = {}
    if not users and os.environ.get("LEMNA_USERS"):
        for item in os.environ["LEMNA_USERS"].split(","):
            if ":" in item:
                name, h = item.split(":", 1)
                users[name.strip().lower()] = h.strip()
    return users


def login_gate(st):
    """Show the login form until a user signs in. Returns the user name, or None if login is off."""
    users = load_users(st.secrets)
    if not users:
        return None
    if st.session_state.get("auth_user") in users:
        return st.session_state.auth_user

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("## 🌿 Lemna Analiz")
        st.caption("Bu uygulama yalnızca proje ekibine açıktır.")
        locked_until = st.session_state.get("auth_locked_until", 0)
        if time.time() < locked_until:
            st.error(f"Çok sayıda hatalı deneme. {int(locked_until - time.time()) + 1} saniye sonra tekrar deneyin.")
            st.stop()
        with st.form("login"):
            name = st.text_input("Kullanıcı adı")
            password = st.text_input("Şifre", type="password")
            ok = st.form_submit_button("Giriş yap", type="primary", width="stretch")
        if ok:
            key = name.strip().lower()
            if key in users and verify_password(password, users[key]):
                st.session_state.auth_user = key
                st.session_state.auth_fails = 0
                st.rerun()
            fails = st.session_state.get("auth_fails", 0) + 1
            st.session_state.auth_fails = fails
            time.sleep(1)
            if fails >= MAX_FAILS:
                st.session_state.auth_fails = 0
                st.session_state.auth_locked_until = time.time() + LOCK_SECONDS
            st.error("Kullanıcı adı veya şifre hatalı.")
    st.stop()


def logout(st):
    for k in ["auth_user", "auth_fails", "auth_locked_until"]:
        st.session_state.pop(k, None)


if __name__ == "__main__":
    name = input("Kullanıcı adı: ").strip().lower()
    pw = getpass.getpass("Şifre: ")
    if len(pw) < 8:
        raise SystemExit("Şifre en az 8 karakter olmalıdır.")
    if pw != getpass.getpass("Şifre (tekrar): "):
        raise SystemExit("Şifreler aynı değil.")
    print("\n.streamlit/secrets.toml dosyasındaki [users] bölümüne şu satırı ekleyin:\n")
    print(f'{name} = "{hash_password(pw)}"')
