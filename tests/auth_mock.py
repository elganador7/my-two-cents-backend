import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.utils import to_base64url_uint


def generate_public_private_key_pair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return (public_key, private_key)


(public_key, private_key) = generate_public_private_key_pair()

ALGORITHM = "RS256"
PUBLIC_KEY_ID = "sample-key-id"


def encode_token(payload):
    return jwt.encode(
        payload=payload,
        key=private_key,  # The private key created in the previous step
        algorithm=ALGORITHM,
        headers={
            "kid": PUBLIC_KEY_ID,
        },
    )


def get_mock_user_claims(permissions):
    return {
        "sub": "123|auth0",
        "iss": "auth0_domain",  # Should match the issuer your app expects
        "aud": "api_audience",  # match the audience your app expects
        "iat": 0,  # Issued a long time ago: 1/1/1970
        "exp": 9999999999,  # One long-lasting token, expiring 11/20/2286
        "permissions": permissions,
    }


def get_mock_token(permissions):
    return encode_token(get_mock_user_claims(permissions))


def get_mock_jwk(public_key):
    public_numbers = public_key.public_numbers()

    return {
        "keys": [
            {
                "kid": PUBLIC_KEY_ID,  # Public key id constant from previous step
                "alg": ALGORITHM,  # Algorithm constant from previous step
                "kty": "RSA",
                "use": "sig",
                "n": to_base64url_uint(public_numbers.n).decode("ascii"),
                "e": to_base64url_uint(public_numbers.e).decode("ascii"),
            }
        ]
    }


mock_jwk = get_mock_jwk(public_key)
