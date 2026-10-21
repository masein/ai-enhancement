"""26: Cloudflare R2, read-only, through its S3 API — a listing and a file,
each request signed with AWS Signature Version 4 (the board's image has no
boto3, and two GETs need none). Its keys come from the server's .env alone;
nothing here writes to the bucket, and an error says what failed and its
code, never the endpoint or a key."""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

REGION = "auto"                         # R2's
EMPTY = hashlib.sha256(b"").hexdigest()
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
TIMEOUT = 60


class R2Error(RuntimeError):
    pass


def _q(s: str, safe: str = "-_.~") -> str:
    return urllib.parse.quote(s, safe=safe)


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def sign(method: str, url: str, key_id: str, secret: str, *, headers: dict | None = None,
         payload_hash: str = EMPTY, region: str = REGION,
         now: _dt.datetime | None = None) -> dict[str, str]:
    """the request's headers, signed (AWS SigV4, service s3, path-style)"""
    u = urllib.parse.urlsplit(url)
    amz = (now or _dt.datetime.now(_dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    hdrs = {**{k.lower(): str(v) for k, v in (headers or {}).items()}, "host": u.netloc,
            "x-amz-date": amz, "x-amz-content-sha256": payload_hash}
    signed = ";".join(sorted(hdrs))
    canon = "".join(f"{k}:{' '.join(hdrs[k].split())}\n" for k in sorted(hdrs))
    path = _q(urllib.parse.unquote(u.path) or "/", safe="-_.~/")
    query = "&".join(f"{_q(k)}={_q(v)}" for k, v in
                     sorted(urllib.parse.parse_qsl(u.query, keep_blank_values=True)))
    creq = "\n".join([method, path, query, canon, signed, payload_hash])
    scope = f"{amz[:8]}/{region}/s3/aws4_request"
    sts = "\n".join(["AWS4-HMAC-SHA256", amz, scope, hashlib.sha256(creq.encode()).hexdigest()])
    k = _hmac(("AWS4" + secret).encode(), amz[:8])
    for part in (region, "s3", "aws4_request"):
        k = _hmac(k, part)
    sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
    hdrs["authorization"] = (f"AWS4-HMAC-SHA256 Credential={key_id}/{scope}, "
                             f"SignedHeaders={signed}, Signature={sig}")
    return hdrs


class Bucket:
    """one bucket, read with the keys given (never kept anywhere but here)"""

    def __init__(self, endpoint: str, key_id: str, secret: str, bucket: str):
        self.base = endpoint.rstrip("/") + "/" + _q(bucket, safe="-_.~")
        self._id, self._secret = key_id, secret
        self._host = urllib.parse.urlsplit(endpoint).hostname or ""

    def _scrub(self, text: str) -> str:
        """an error's words without the endpoint (its account id) or a key"""
        for s in (self._host, self._id, self._secret):
            if s:
                text = text.replace(s, "…")
        return text

    def _get(self, url: str, what: str) -> bytes:
        req = urllib.request.Request(url, headers=sign("GET", url, self._id, self._secret))
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            code = ""
            try:
                code = (ET.fromstring(e.read() or b"<x/>").findtext("Code") or "")
            except ET.ParseError:
                pass
            raise R2Error(f"{what} answered {e.code}" + (f" ({code})" if code else "")) from None
        except urllib.error.URLError as e:
            raise R2Error(f"{what}: the bucket couldn't be reached "
                          f"({self._scrub(str(e.reason))})") from None
        except TimeoutError:
            raise R2Error(f"{what}: no answer in {TIMEOUT} s") from None

    def keys(self, prefix: str) -> list[str]:
        """every object's key under the prefix (ListObjectsV2, page by page)"""
        out: list[str] = []
        token = None
        while True:
            q = {"list-type": "2", "prefix": prefix, **({"continuation-token": token} if token else {})}
            url = self.base + "?" + urllib.parse.urlencode(q, quote_via=urllib.parse.quote)
            root = ET.fromstring(self._get(url, "the listing"))
            out += [k.text for k in root.iter(NS + "Key") if k.text]
            if (root.findtext(NS + "IsTruncated") or "").lower() != "true":
                return out
            token = root.findtext(NS + "NextContinuationToken")
            if not token:
                return out

    def get(self, key: str) -> bytes:
        return self._get(self.base + "/" + _q(key, safe="-_.~/"), "a file")
