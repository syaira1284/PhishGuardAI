import re
import ipaddress
from urllib.parse import urlparse
from collections import Counter
from math import log2
import tldextract

def normalize_url(url):
    """
    Membersihkan URL dari skema (http/https) dan 'www.'
    untuk menghilangkan bias pada dataset dan model.
    """
    url = str(url).strip().lower()

    # 1. Hapus http:// atau https://
    url = re.sub(r'^https?:\/\/', '', url)

    # 2. Hapus www.
    url = re.sub(r'^www\.', '', url)

    # 3. Hapus slash '/' di akhir URL (jika tidak ada path lain) agar rapi
    if url.endswith('/') and url.count('/') == 1:
        url = url[:-1]

    return url

def extract_features(url):
    # 1. Normalisasi URL untuk menghilangkan bias
    clean_url = normalize_url(url)

    # 2. Tambahkan dummy scheme 'http://' khusus untuk urlparse
    # (Hanya agar urlparse bisa memisahkan path/query dengan benar, tidak dihitung fiturnya)
    parsed = urlparse('http://' + clean_url)

    # 3. Ekstrak domain
    ext = tldextract.extract(clean_url)
    domain = ext.domain
    suffix = ext.suffix

    # 4. Fitur berbasis panjang dan karakter (menggunakan URL yang sudah bersih)
    url_length = len(clean_url)
    letters = sum(c.isalpha() for c in clean_url)
    digits = sum(c.isdigit() for c in clean_url)

    # 5. Fitur Special Characters
    NORMAL_URL_CHARS = set(':/.?=&-_#@%+~')
    path_and_query = parsed.path + ('?' + parsed.query if parsed.query else '')

    special_chars = sum(
        c not in NORMAL_URL_CHARS and not c.isalnum()
        for c in path_and_query
    )

    # 6. Fitur Subdomain
    # Karena 'www' sudah dihapus di normalize_url, subdomain akan dihitung dengan adil
    if ext.subdomain:
        subs = ext.subdomain.split(".")
        subdomains = len(subs)
    else:
        subdomains = 0

    # 7. Cek IP Address
    is_domain_ip = 0
    try:
        if ext.domain:
            ipaddress.ip_address(ext.domain)
            is_domain_ip = 1
    except ValueError:
        pass

    ip_pattern = r"(?:\d{1,3}\.){3}\d{1,3}"
    if re.search(ip_pattern, clean_url):
        is_domain_ip = 1

    # 8. Fitur Obfuscation
    encoded_patterns = re.findall(r"%[0-9A-Fa-f]{2}", clean_url)
    no_of_obfuscated_char = len(encoded_patterns)
    has_obfuscation = 1 if no_of_obfuscated_char > 0 else 0
    obfuscation_ratio = (no_of_obfuscated_char / url_length) if url_length > 0 else 0

    # 9. Fitur Hyphen & Suspicious Words
    hyphen_count = clean_url.count("-")

    suspicious_words = [
        "login", "secure", "verify", "update", "account",
        "signin", "banking", "confirm", "password"
    ]
    suspicious_word_count = sum(word in clean_url for word in suspicious_words)

    # 10. Fitur Entropi
    entropy = 0
    if len(domain) > 0:
        char_counts = Counter(domain)
        probabilities = [count / len(domain) for count in char_counts.values()]
        entropy = -sum(p * log2(p) for p in probabilities)

    # Return sesuai format DataFrame yang dibutuhkan model
    return {
        "URLLength":                  url_length,
        "DomainLength":               len(domain),
        "TLDLength":                  len(suffix),
        "NoOfSubDomain":              subdomains,
        "NoOfLettersInURL":           letters,
        "NoOfDegitsInURL":            digits,
        "DegitRatioInURL":            digits / url_length if url_length > 0 else 0,
        "NoOfOtherSpecialCharsInURL": special_chars,
        "SpacialCharRatioInURL":      special_chars / url_length if url_length > 0 else 0,
        "IsDomainIP":                 is_domain_ip,
        "HasObfuscation":             has_obfuscation,
        "NoOfObfuscatedChar":         no_of_obfuscated_char,
        "ObfuscationRatio":           obfuscation_ratio,
        "HyphenCount":                hyphen_count,
        "SuspiciousWordCount":        suspicious_word_count,
        "Entropy":                    entropy,
    }