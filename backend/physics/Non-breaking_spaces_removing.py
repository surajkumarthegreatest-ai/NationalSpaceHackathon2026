# Read and clean the file
with open("", "r", encoding="utf-8") as file:
    content = file.read()

# Replace non-breaking spaces with regular spaces
cleaned_content = content.replace("\u00A0", " ")

# Write back to the file
with open("", "w", encoding="utf-8") as file:
    file.write(cleaned_content)
print("Non-breaking spaces removed.")