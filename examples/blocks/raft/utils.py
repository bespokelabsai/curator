def extract_text(path, backend="aryn"):
    """Extract text from a PDF file using the specified backend."""
    print(f"Extracting text from {path} using {backend} backend")
    text = []
    if backend == "aryn":
        from aryn_sdk.partition import partition_file

        with open(path, "rb") as f:
            data = partition_file(f)

        elements = data["elements"]
        for element in elements:
            try:
                text.append(element["text_representation"])
            except KeyError:
                continue
    elif backend == "local":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
    else:
        raise ValueError(f"Invalid OCR backend: {backend}")
    return "".join(text)
