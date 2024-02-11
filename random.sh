find / -type f ! -name "*.*" -exec grep -lvIP '[^[:ascii:]]' {} + 2> /dev/null | shuf -n 1200

