import pdfkit

# Path to the HTML file you want to convert
html_file = 'combined.html'

# Output PDF file path
output_pdf = 'output.pdf'

# Path to the wkhtmltopdf executable (if needed)
# Uncomment the following line and specify the path to wkhtmltopdf if it's not in your PATH
path_to_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'  # Example for Windows

# If wkhtmltopdf is in PATH, you don't need to pass the configuration
config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

try:
    # Convert HTML to PDF (uncomment the config line if wkhtmltopdf path is specified)
    # pdfkit.from_file(html_file, output_pdf, configuration=config)  # If using the config
    print("started")
    pdfkit.from_file(html_file, output_pdf)  # If wkhtmltopdf is in PATH
    print(f"PDF generation complete: {output_pdf}")
except Exception as e:
    print(f"Error generating PDF: {e}")
