
# Leer-je-eigen-documenten-bevragen


<!--
LEER RAG IMPLEMENTEREN MET LangChain + Azure + Python + JupyterNotebook + GitHub
-->

<img align="right" width="200" height="200" src="https://avatars.githubusercontent.com/u/115706761?s=400&u=7c6cae892816e172b0b7eef99f2d32adb948c6ad&v=4">

## Context & Doelen

| RAG implementatie met Azure + LangChain + OpenAI |
|-----|
| 1. Begrijpen wat RAG wel en niet kan [[Wat is RAG]](#intro) 
| 2. Veligiheidsmaatregelen nemen 
| 3. LanChain installeren en configureren
| 4. Azure OpenAI API key aanvragen
| 5. Jupyter Notebook installeren
| 6. DEMO [[DEMO]](#demo).

Deze GitHub Repository geeft inzicht hoe je met behulp van  Generatieve-AI (Gen-AI) je eigen documenten kunt bevragen.

>Disclaimer: deze tekst is door het gebruik van *"gezond verstand'* tot stand gekomen. <br> Artificiële intelligentie [AI] is gebruikt ter verificatie van de gebruikte bronnen + vertaling van Engelstalige teksten.

Dit is een data product gemaakt door het [PROMETHEUS DATA SCIENCE LAB](https://github.com/HR-DATA-FABRIC/PROMETHEUS) van de Hogeschool Rotterdam.

  Views since 15 juni 2023: [![HitCount](https://hits.dwyl.com/robvdw/HR-DATA-FABRIC/Leer-je-eigen-documenten-bevragen-met-generatieve-AI.svg?style=flat-square)](http://hits.dwyl.com/robvdw/HR-DATA-FABRIC/Leer-je-eigen-documenten-bevragen-met-generatieve-AI)
  <br>
  Unique visitors since 15 juni 2023: [![HitCount](https://hits.dwyl.com/robvdw/HR-DATA-FABRIC/Leer-je-eigen-documenten-bevragen-met-generatieve-AI.svg?style=flat-square&show=unique)](http://hits.dwyl.com/robvdw/HR-DATA-FABRIC/Leer-je-eigen-documenten-bevragen-met-generatieve-AI)


***********
# Context
***********
## Waarom is RAG nodig als ik al vragen kan stellen over teksten via Gen-AI? 

Gen-AI applicaties gebaseerd op grote taalmodellen (LLMs) verwijzen naar elke vorm van machinaal-lerende (ML) kunstmatige intelligentie (AI) die gebruikt maakt van natuurlijke taal verwerkende (NLP) algoritmen. 

Eindgebruikers  kunnen  online gebruik  maken deze conversationele agenten (Chatbots zoals Bard, Co-Pilot, en ChatGPT) via webbrowser userinterfaces.  Hierdoor is het mogelijk om via zelf-geschreven, textuele instructies (zogenaamde prompts), content te genereren, in de vorm van tekst, broncode, afbeeldingen, video's, muziek etc.

LLMs worden tijdens hun trainingfase gevoed met publiekelijk beschikbare content, in de vorm van boeken, video's, audio opnames, databases, artikelen, websites, en open source-broncode. Zo leren LLMs output te creëren die sterk lijkt op door mensen gemaakte, authentieke content. Op het moment dat eindgebruikers gebruik maken van op LLM-gebaseerde gen-AI userinterfaces dan is het onderliggende taal model bevroren en kan niet meer worden getraind.  De Engelse term “pre-trained” wordt veelal gebruikt om dit aan te geven. Het onderliggende taalmodel van de ChatGPT interface is GPT-4 hetgeen staat voor: 4de generatie “Generative Pre-trained Transformer”.

Een groot nadeel van de huidige generatie LLMs is dat ze feitelijk functioneren als algemene (lees general-purpose) kennisbanken, waarvan de hoeveelheid beschikbare gegevens niet uitbreidbaar (lees, bevroren) en niet op één enkele locatie in het model is opgeslagen, maar over verschillende locaties is verspreid (lees, gedistribueerd).
Met andere woorden, de in LLMs opgeslagen kennis is voor eindgebruikers zowel onveranderbaar als ook  onherkenbaar waardoor taalmodellen niet eenvoudig doelgericht zijn te bevragen over een specifiek en actueel kennisdomein. Het is dus  niet mogelijk om te bepalen welke informatie het LLM  benut om antwoorden te genereren, waardoor waarheidsvinding (fact-checking) moeilijk uitvoerbaar is.

Deze tekortkomingen kunnen worden tegengegaan door gebruik te maken van Retrieval-Augmented Generation (RAG). Deze op natuurlijke taal generatie (NLP) gebaseerde AI-technologie kun je opvatten als een onderzoeks- en schrijversduo. Stel je voor dat je een journalist bent die verslag doet van  een natuurramp. Je doet dan eerst onderzoek naar de gebeurtenis, verzamelt relevante artikelen of weer rapporten en gebruikt deze informatie om dit specifieke nieuwsverhaal te schrijven.

RAG doet iets soortgelijks maar dan voor grote-taalmodellen. De retriever-component representeert de journalist die relevante informatie verzamelt, en de generator-component is de schrijver die deze informatie gebruikt om een voor mensen begrijpelijke en waardevolle nieuwsverhaal te schrijven.


***********
# demo

Deze demo is gebasserd op de onderstaande bron:
* https://learn.microsoft.com/en-us/azure/ai-services/openai/use-your-data-quickstart?tabs=command-line%2Cpython&pivots=programming-language-python
***********
## RAG implementatie met Azure + LangChain + OpenAI

### Stap 1 

#### Python Package Installatie via Jupyter NoteBook use Colab or Anaconda

````Python
# Installeer de benodigde packages via Jupyter Notebook
import sys
!{sys.executable} -m pip install python-dotenv langchain unstructured[pdf] openai==0.28.1 chromadb tiktoken
  ````

### Stap 2

#### Importeer de benodigde LangChain modules
````python
from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
  ````
##### Referenties
* LangChain Startpagina   ====> https://python.langchain.com/docs/get_started/introduction
* LangChain Azure OpenAI  ====> https://python.langchain.com/docs/integrations/llms/azure_openai


### Stap 3

Wanneer je applicatie zijn configuratie ontleent aan omgevingsvariabelen (system variables), 
dan kun je dotenv aan je applicatie toevoegen zodat het de benodigde variablen uit een .env bestand kunt uitlezen.


````python
# de benodigde Azure deployment information is stored in a .env file
# in de zelfde directory als  het notebook

# omgevings variabelen geschikt in combionatie met OpenAI 0.28.1 package 
'''
# Set this to `azure`
OPENAI_API_TYPE =     "azure"

# The API version you want to use: set this to `2023-05-15` for the released version.
OPENAI_API_VERSION =  "xxxxxx"

# The base URL for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
OPENAI_API_BASE =     "https://xxxxxx.openai.azure.com/"

# The API key for your Azure OpenAI resource.  Select one of the deployments from the deployment history.
OPENAI_API_KEY =      "xxxxxx"

# The name of your Azure OpenAI deployment.  You can find this in the Azure portal under your Azure OpenAI resource.
DEPLOYMENT_NAME =     "xxxxxx"
'''

import os
from io import StringIO
from dotenv import load_dotenv
load_dotenv(override=True)
  ````
  ##### Referenties
  * Wat zijn omgevings variabelen.   ====> https://geekflare.com/nl/python-environment-variables/
  * Dotenv installeer pagina.         ====> https://github.com/theskumar/python-dotenv

### Stap 4

De UnstructuredFileLoader ondersteunt het laden van vele bestandstypes zoals PDF's, PPT's, afbeeldingen, enz.

````python
# from langchain.document_loaders import UnstructuredFileLoader
# ====> laad een document "sample.pdf' in de variabele "documents"
# Partitioning Strategy: "fast" or "accurate"
# Partitioning Mode: "single", "elements", or "paged
# single ===> all the text from all elements are combined into one (default)
# elements ===> maintain individual elements
# paged ===> texts from each page are only combined


loader = UnstructuredFileLoader('Sample.pdf', strategy='fast')
documents = loader.load()

# toon de inhoud van het "sample.pdf" document
display(documents)
  ````
Referenties
* Uitleg langchain.document_loaders. =====> https://python.langchain.com/docs/integrations/document_loaders/unstructured_file