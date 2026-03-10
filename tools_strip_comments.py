import os
import io
import re
import tokenize

ROOT = os.path.dirname(__file__)

def strip_py_comments(src: str) -> str:
    buf = io.StringIO(src)
    tokens = list(tokenize.generate_tokens(buf.readline))
    out_tokens = []
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            continue
        out_tokens.append(tok)
    return tokenize.untokenize(out_tokens)

def strip_html_comments(src: str) -> str:
    return re.sub(r'<!--[\s\S]*?-->', '', src)

def strip_css_comments(src: str) -> str:
    return re.sub(r'/\*[\s\S]*?\*/', '', src)

def process_file(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            src = f.read()
        if path.endswith('.py'):
            new_src = strip_py_comments(src)
        elif path.endswith('.html'):
            new_src = strip_html_comments(src)
        elif path.endswith('.css'):
            new_src = strip_css_comments(src)
        else:
            return
        if new_src != src:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_src)
    except Exception:
        pass

def main():
    for root, dirs, files in os.walk(ROOT):
        # skip venv or data folders if present
        if any(sk in root for sk in ['.venv', 'venv', '__pycache__']):
            continue
        for name in files:
            if name.endswith(('.py', '.html', '.css')):
                process_file(os.path.join(root, name))

if __name__ == '__main__':
    main()
