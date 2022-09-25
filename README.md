# Project name

## Team Members
Ryan Willett

## Tool Description
NERDoc is a framework for analyzing documents. The goal of the project, which is currently unfinished, is to address the needs of OSINT analysts to process either large documents or large collections of documents and extract meaning from them.

The MVP functionality will include:
- OCR documents (single and batch)
- Named Entity Recognition (NER) on websites and documents
    - This automatically extracts features from the document for use by the viewer, including:
        - People
        - Organizations
        - Places (countries, regions)
        - Phrases concerning money
        - Nationalities, Religions
    - Batched NER analysis allows one to construct network graphs connecting the NERs of various documents
- Extractive summarization (single and batch documents)
## Installation

Installation of 3rd party dependencies will be automated with a bash script in future versions

1. Download the code from the project repo
        git clone git@github.com:rtwillett/NERDoc.git

2. Install Tesseract

    On Mac: 
        If you have homebrew
            brew install tesseract

    On Window: This installation has not been tested or detailed in this version

3. Install poppler

    On Mac: 
        If you have homebrew
            brew install poppler

4. Install the requirements for the project

    Create a virtual environment for the project
    python3 -m venv env

    Activate the environment
    On Mac
        source env/bin/activate

    pip install the dependencies

    pip install -r requirements.txt

5. Install the large language model for spaCy trained on web text data

    python -m spacy download en_core_web_lg

## Usage

Deploy the application by navigating to the project directory and running

python app.py

The app may also be configured by setting the PATH
    export FLASK_APP=app.py
    flask run

This is a web application and will likely be deployed to the web in future iterations but was beyond the scope of the hackathon

## Additional Information
This section includes any additional information that you want to mention about the tool, including:
- Potential next steps for the tool (i.e. what you would implement if you had more time)
- Any limitations of the current implementation of the tool
- Motivation for design/architecture decisions

### Next Steps
- Getting single document functionlity online
- Getting batch functionality online
- At the moment, only single document OCR actually works