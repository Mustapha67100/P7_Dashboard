mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = ${PORT:-8042}\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml