# URL Feature Dictionary

All 26 model features are computed locally from the submitted URL string. No URL
is opened, redirected, DNS-resolved, or enriched with WHOIS/reputation data when
extracting them. (Domain age, DNS and TLS data are gathered separately by
`src/domain_intelligence.py` and applied after the model; they are not model
features. See README → Domain intelligence.)

| Feature | Description and motivation |
| --- | --- |
| `url_length` | Character count after deterministic scheme normalization; unusually long URLs can conceal a destination. |
| `domain_length` | Hostname character count. |
| `path_length` | Path character count. |
| `query_length` | Query-string character count. |
| `dot_count` | Dot count across the normalized URL. |
| `hyphen_count` | Hyphen count across the normalized URL. |
| `underscore_count` | Underscore count across the normalized URL. |
| `digit_count` | Number of digits across the normalized URL. |
| `special_char_count` | Count of URL punctuation characters. |
| `subdomain_count` | Host labels before a small registered-suffix heuristic. |
| `query_param_count` | Non-empty ampersand-separated query fields. |
| `at_count` | `@` count; it can obscure the apparent authority in a URL. |
| `equals_count` | `=` count. |
| `ampersand_count` | `&` count. |
| `percent_count` | `%` count, indicating percent-encoded content. |
| `colon_count` | `:` count, including scheme and any explicit port. |
| `suspicious_keyword_count` | Occurrences of a documented fixed lexical keyword list. |
| `digit_ratio` | Digits divided by URL length. |
| `special_char_ratio` | URL punctuation divided by URL length. |
| `url_entropy` | Shannon entropy (bits/character) of the normalized URL. |
| `uses_https` | One when the explicit/effective scheme is HTTPS. |
| `ip_in_url` | One when the hostname parses as IPv4 or IPv6. |
| `suspicious_tld` | One when the TLD belongs to a small documented heuristic list; it is not a reputation verdict. |
| `domain_path_ratio` | Hostname length divided by at least one path character. |
| `has_punycode` | One when `xn--` appears in the hostname. |
| `has_port` | One when the URL has an explicit port. |

Feature groups used in the ablation study are basic lexical, structural/security,
entropy, and suspicious-keyword features. The implementation preserves the
original project's useful length/count/HTTPS/IP/keyword/entropy/path/query
features and makes extraction identical in training and serving.
