mkdir -p ~/.streamlit/

echo "\"
[serveur]\n\
headless = true\n\
port = $PORT\n\
enableeCORS = false\n\
\n\
" > ~/.streamlit/config.toml