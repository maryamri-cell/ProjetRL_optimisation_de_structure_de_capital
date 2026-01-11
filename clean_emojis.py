import os
import re
from pathlib import Path

# Emoji pattern (covers most unicode emoji characters)
emoji_pattern = re.compile(
    "["
    "\U0001F300-\U0001F9FF"  # Emoticons, Symbols, Pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\u2600-\u26FF"          # Miscellaneous Symbols
    "\u2700-\u27BF"          # Dingbats
    "\u2300-\u23FF"          # Miscellaneous Technical
    "\u2000-\u200D"          # General Punctuation
    "\u2010-\u205E"          # General Punctuation
    "\u2070-\u209F"          # Superscripts and Subscripts
    "\u20A0-\u20CF"          # Currency Symbols
    "\u20D0-\u20FF"          # Combining Diacritical Marks
    "\u2100-\u214F"          # Letterlike Symbols
    "\u2150-\u218F"          # Number Forms
    "\u2190-\u27FF"          # Arrows, Mathematical Operators, Technical Symbols
    "\u2800-\u28FF"          # Braille Patterns
    "\u2900-\u297F"          # Supplemental Arrows-B
    "\u2B00-\u2BFF"          # Miscellaneous Symbols and Arrows
    "\u2E00-\u2E7F"          # Supplemental Punctuation
    "\u3000-\u303F"          # CJK Symbols and Punctuation
    "\uA960-\uA97F"          # Hangul Jamo Extended-A
    "\uFE00-\uFE0F"          # Variation Selectors
    "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "✅❌⭐💡🚀📈🎨🔍📱🎯📚📊🔧🖼️📄🐍✔️"
    "]+"
)

doc_path = Path('documentation')
count = 0

for md_file in doc_path.glob('**/*.md'):
    if md_file.name == 'README.md' or md_file.name.startswith(('0', '1')):
        with md_file.open('r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        content = emoji_pattern.sub('', content)
        
        if content != original:
            # Clean up extra spaces left by emoji removal
            content = re.sub(r' +\n', '\n', content)
            content = re.sub(r'\n\n\n+', '\n\n', content)
            
            with md_file.open('w', encoding='utf-8') as f:
                f.write(content)
            count += 1
            print(f'✓ {md_file.name}')

print(f'\nTotal files cleaned: {count}')
