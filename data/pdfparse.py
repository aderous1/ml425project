import PyPDF2
import re
import json

pdf = open('116th-Congress-Twitter-Handles.pdf', 'rb')

reader = PyPDF2.PdfFileReader(pdf)

pages = reader.numPages

text = ""
handles = []

for i in range(0,pages):
    pageObj = reader.getPage(i)
    text += pageObj.extractText()

pdf.close()


for line in text.splitlines():
    if ('@' in line):
        handle = re.search("@[^ ]*", line)
        if(handle):
            handles.append(handle[0])
        

y = json.dumps(handles)
print(y)