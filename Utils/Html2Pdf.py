import requests
from bs4 import BeautifulSoup
import pdfkit

# Function to get the HTML content of a webpage
def get_page_content(url):
    try:
        response = requests.get(url, timeout=10)  # Set a timeout to avoid long waits
        if response.status_code == 200:
            return response.content
        else:
            print(f"Failed to download {url} - Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

# URL of the main page
main_url = 'https://docs.moodle.org/403/en/Category:Quick_guide'

# Get the content of the main page
main_page_content = get_page_content(main_url)
if not main_page_content:
    print(f"Failed to download {main_url}")
    exit()

# Parse the main page HTML
soup = BeautifulSoup(main_page_content, 'html.parser')

# Prepare the combined HTML content
combined_html_content = '<h1>Main Page Content</h1>'
combined_html_content += str(soup)  # Add the main page content

# Find all hyperlinks in the main page
links = soup.find_all('a', href=True)

# Keep track of visited URLs to avoid duplicates
visited_links = set()

# Loop through each hyperlink, download its content, and add it to the combined content
for link in links:
    href = link['href']
    # Ensure that we have a valid URL (absolute URL)
    if not href.startswith('http'):
        href = requests.compat.urljoin(main_url, href)
    
    if href not in visited_links:
        print(f"Processing: {href}")
        try:
            page_content = get_page_content(href)
            if page_content:
                # Parse the linked page and add its content
                link_soup = BeautifulSoup(page_content, 'html.parser')
                combined_html_content += f'<h1>{link.text}</h1>'  # Add a heading for the link
                combined_html_content += str(link_soup)  # Add the linked page content
            
            visited_links.add(href)
        
        except Exception as e:
            print(f"Error processing {href}: {e}")
            # Skip to the next link if there's an error

# Save the combined content to an HTML file
with open('combined.html', 'w',encoding='utf-8') as file:
    file.write(combined_html_content)

# Convert the combined HTML to PDF
try:
    pdfkit.from_file('combined.html', 'output.pdf')
    print("PDF generation complete: output.pdf")
except Exception as e:
    print(f"Error generating PDF: {e}")
