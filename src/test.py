from pdf_parser import parse_pdf
df = parse_pdf("data/Statement.pdf")
print(len(df)); print(df.head(8)); print(df.tail(3))