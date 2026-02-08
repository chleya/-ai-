import markdown

md = open('F:/system_stability/ROBUSTNESS_FRAMEWORK_FINAL.md', 'r', encoding='utf-8').read()

html = '''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>边缘自主演化群的鲁棒性框架</title>
<style>
body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.8; font-size: 14px; }
h1 { font-size: 22px; text-align: center; color: #333; }
h2 { font-size: 16px; margin-top: 25px; color: #444; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
h3 { font-size: 14px; color: #555; margin-top: 18px; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 8px; }
th { background-color: #f2f2f2; }
pre { background-color: #f5f5f5; padding: 10px; overflow-x: auto; }
code { background-color: #f5f5f5; padding: 2px 4px; }
blockquote { border-left: 4px solid #ddd; margin: 15px 0; padding-left: 15px; color: #666; }
</style>
</head>
<body>
<h1>边缘自主演化群的鲁棒性框架</h1>
'''

html += markdown.markdown(md, extensions=['tables'])

html += '''
<hr>
<p style="color: #888; font-size: 12px; text-align: center;">
Generated: 2026-02-07 | F:/system_stability/ROBUSTNESS_FRAMEWORK_FINAL.md
</p>
</body>
</html>'''

with open('F:/system_stability/report.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('HTML generated: F:/system_stability/report.html')
