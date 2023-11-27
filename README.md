
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
| 3. LangChain leren gebruiken voor RAG implementatie met 
| 4. Azure OpenAI API key + deployment aanmaken voor model: "text-embedding-ada-002"
| 5. Jupyter Notebook aanmaken in CoLab of Anaconda
| 6. DEMO [[DEMO]](#demo).

Deze GitHub Repository geeft inzicht hoe je met behulp van  Generatieve-AI (Gen-AI) je eigen documenten kunt bevragen.

>Disclaimer: deze tekst is door het gebruik van *"gezond verstand'* tot stand gekomen. <br> Artificiële intelligentie [AI] is gebruikt ter verificatie van de gebruikte bronnen + vertaling van Engelstalige teksten.

Dit is een data product gemaakt door het [PROMETHEUS DATA SCIENCE LAB](https://github.com/HR-DATA-FABRIC/PROMETHEUS) van de Hogeschool Rotterdam.

  Views since 23 november 2023: [![HitCount](https://hits.dwyl.com/robvdw/HR-DATA-FABRIC/Leer-je-eigen-documenten-bevragen-met-generatieve-AI.svg?style=flat-square)](http://hits.dwyl.com/robvdw/HR-DATA-FABRIC/Leer-je-eigen-documenten-bevragen-met-generatieve-AI)
  <br>
  Unique visitors since 23 november 2023: [![HitCount](https://hits.dwyl.com/robvdw/HR-DATA-FABRIC/Leer-je-eigen-documenten-bevragen-met-generatieve-AI.svg?style=flat-square&show=unique)](http://hits.dwyl.com/robvdw/HR-DATA-FABRIC/Leer-je-eigen-documenten-bevragen-met-generatieve-AI)


***********
# Context & waarde voor het hoger onderwijs
***********
## Waarom is RAG nodig als ik al vragen kan stellen over teksten via Gen-AI? 

Gen-AI applicaties gebaseerd op grote taalmodellen (LLMs) is elke vorm van machinaal-lerende (ML) kunstmatige intelligentie (AI) die gebruikt maakt van natuurlijke taal verwerkende (NLP) algoritmen. 

Eindgebruikers  kunnen  online gebruik  maken deze conversationele agenten (Chatbots zoals Bard, Co-Pilot, en ChatGPT) via webbrowser userinterfaces.  Hierdoor is het mogelijk om via een toetsenbord ingevoerde instructies (zogenaamde prompts), content te genereren, in de vorm van tekst, broncode, afbeeldingen, video's, muziek etc.

LLMs worden tijdens hun trainingfase gevoed met publiekelijk beschikbare content, in de vorm van boeken, video's, audio opnames, databases, artikelen, websites, en open source-broncode. Zo leren LLMs output te creëren die sterk lijkt op door mensen gemaakte, authentieke content. Op het moment dat eindgebruikers een vraag stelt in de vorm van een prompt dan is het onderliggende taal model bevroren en kan niet meer worden getraind.  De Engelse term “pre-trained” wordt veelal gebruikt om dit aan te geven. Het onderliggende taalmodel van de ChatGPT interface is GPT-4 hetgeen staat voor: 4de generatie “Generative Pre-trained Transformer”.

Een groot nadeel van de huidige generatie LLMs is dat ze feitelijk functioneren als algemene (lees general-purpose) kennisbanken, waarvan de hoeveelheid beschikbare gegevens niet uitbreidbaar (lees, bevroren) en niet op één enkele locatie in het model is opgeslagen, maar over verschillende locaties is verspreid (lees, gedistribueerd). Dus, de in LLMs opgeslagen kennis is voor eindgebruikers zowel onveranderbaar als ook  onherkenbaar waardoor taalmodellen niet eenvoudig doelgericht zijn te bevragen over een specifiek en actueel kennisdomein. Het is dan ook niet mogelijk om te bepalen welke informatie het LLM  benut om antwoorden te genereren, waardoor waarheidsvinding (fact-checking) moeilijk uitvoerbaar is.

Deze tekortkomingen kunnen worden gecompenseerd door ["Retrieval-Augmented Generation"](https://doi.org/10.48550/arXiv.2005.11401) (RAG). Deze op natuurlijke taal generatie (NLP) gebaseerde AI-technologie kan externe digitale-bronnen (zoals API's, Websites, SQL-databases, PDF en Excel  files) koppelen aan LLMs. Hierdoor wordt de informatie die is opgeslagen in de externe bron direct opvraagbaar via prompts. De werking van RAG is op te vatten als de combinatie van een kundige schrijver (het LLM) die alle externe informatie kan uitleggen die hem door de bibliothecaris (de database) is verstrekt. 

De combinatie van een LLM met een externe database is een vorm van “hybride AI” waarbij de gegevens van een LLM worden gecombineerd met de kennis van een database. RAG vergroot dus de zeer brede kennisbasis van een LLM met betrouwbare en zeer specifieke informatie. Dus met RAG is het mogelijk om een "fact-checking" component toe te voegen aan een LLM.

Een voor de hand liggende toepassing is dan ook het bevraagbaar maken van intranet websites, databases, en handleidingen. Hierdoor kunnen eindgebruikers via een webbrowser interface, in de vorm van een chatbot, vragen stellen over de inhoud van deze digitale bronnen.
In het hoger onderwijs kan RAG worden ingezet om studenten te helpen bij het vinden van informatie in de vorm van antwoorden op vragen over het LMS (Learning Management System) zoals Canvas, Brightspace, Blackboard, enz. Verder kan RAG worden ingezet om patronen te ontdekken in de antwoorden die studenten geven op open vragen in toetsen. Hierdoor kan de kwaliteit van de toetsvragen worden verbeterd. Ook kan RAG worden benut om studenten te helpen bij het vinden van informatie in de vorm van antwoorden op vragen over de digitale bibliotheek, de digitale leeromgeving, enz.


### Retrieval versus Generatie https://python.langchain.com/docs/use_cases/question_answering/
<img align="center" width="1000" height="400" src=".\rag_retrieval_generation.png">

### Geraadplegde bronnen:
* [Quickstart: Chat with Azure OpenAI models using your own data](https://learn.microsoft.com/en-us/azure/ai-services/openai/use-your-data-quickstart?tabs=command-line%2Cpython&pivots=programming-language-python)
* [LangChain: What is RAG?](https://python.langchain.com/docs/use_cases/question_answering/)
* [RAG topassen via Google Colab](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/use_cases/question_answering/index.ipynb)
* [Retrieval Augmented Generation using Azure Machine Learning prompt flow (preview)](https://learn.microsoft.com/en-us/azure/machine-learning/concept-retrieval-augmented-generation?view=azureml-api-2#why-use-rag)
*******


***********
# Veiligheids-issues bij het gebruik van RAG <br>(deze sectie is nog in ontwikkeling)
***********
## Wanneer is RAG veilig te gebruiken?

Met de lancering van GPT-4 Turbo en de Retrieval API heeft OpenAI geprobeerd het hallucinatieprobleem op te lossen. Met de lange contextlengte en de optie voor bedrijven om nieuwe gegevens te integreren voor informatie, heeft OpenAI bijna het belangrijkste probleem van LLM's gekraakt en opgelost, maar de gegevensprivacy van gebruikers vergeten.

Zo kon een gebruiker op X met een beetje betere prompt engineering de originele kennisbestanden downloaden van de GPT's van iemand anders, een app die was gebouwd met de onlangs uitgebrachte GPT Builder, precies met RAG. Dit is een groot beveiligingsprobleem voor dit model.
###  Geraadpleegde bronnen:
* https://analyticsindiamag.com/when-and-when-not-to-rag/

<!--
#### Retriever-component: verzamel + embed relevante informatie
<img align="center" width="1000" height="400" src=".\rag_visual-explaining.png">
-->



<br>


***********
## RAG implementatie met Azure + LangChain + OpenAI

Onderstaande stappen beschrijven hoe je een RAG implementatie kunt realiseren met behulp van Azure + LangChain + OpenAI in een Jupyter Notebook.
Deze stappen zijn getest in een standaard Google Colab Notebook.

### Stap 1 

#### Python Package Installatie via Jupyter NoteBook

````Python
# Installeer de benodigde packages via Jupyter Notebook
import sys
!{sys.executable} -m pip install python-dotenv langchain unstructured[pdf] openai==0.28.1 chromadb tiktoken
  ````

### Stap 2

#### Importeer de benodigde LangChain Python modules
````python
from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
  ````
#### Referenties
* LangChain Startpagina   ====> https://python.langchain.com/docs/get_started/introduction
* LangChain Azure OpenAI  ====> https://python.langchain.com/docs/integrations/llms/azure_openai


### Stap 3

Om toegang te krijgen tot OpenAI via Azure moeten  er gegevens zoals azure-api-key + de deployment naam beschikbaar zijn als systeem variabelen. 
Dit kan op verschillende manieren worden gedaan. Een veilige manier is om de benodigde variabelen in een .env bestand te zetten. 
Dit voorkomt dat de gegevens per ongeluk worden gedeeld via code die is gepubliceerd via GitHub of Google Colab.
Het .env bestand moet dan in de zelfde directory staan als het notebook. 

Voor het aanmaken van azure api keys en deployment namen 
raadpleeg de volgende Github Repository:<br> https://github.com/HR-ChatGPT/ChatGPT-UITGELEGD/tree/main/WORKSHOPS/AZURE.
<br>

````python
'''
# de benodigde Azure deployment information moet zijn opgeslagen in een .env file
# in de zelfde directory als  het notebook

# omgevingsvariabelen geschikt in combionatie met OpenAI 0.28.1 package 

# Set this to `azure`
OPENAI_API_TYPE =     "azure"

# The API version you want to use: set this to `2023-05-15` for the released version.
OPENAI_API_VERSION =  "xxxxxx"

# The base URL for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
OPENAI_API_BASE =     "https://xxxxxx.openai.azure.com/"

# The API key for your Azure OpenAI resource.  Select one of the deployments from the deployment history.
AZURE_OPENAI_API_KEY =      "xxxxxx"

# The name of your Azure OpenAI deployment.  You can find this in the Azure portal under your Azure OpenAI resource.
DEPLOYMENT_NAME =     "xxxxxx"

===>
N.B.  To make embedding work with OPENAI via Azure:
"engine" should be set to the deployment name 
you chose when you deployed the <text-embedding-ada-002 (Version 2)> model
<===
'''

import os
from io import StringIO
from dotenv import load_dotenv
load_dotenv(override=True)
  ````

Een voorbeeld van een werkende embeding configuratie in Azure met OpenAI 0.28.1 package is:
```yaml
https://taalmodel01.openai.azure.com//openai/deployments?api-version=2022-12-01
{
  "data": [
    {
      "scale_settings": {
        "scale_type": "standard"
      },
      "model": "gpt-35-turbo",
      "owner": "organization-owner",
      "id": "DAVINCI",  // <=== engine: deployment Name
      "status": "succeeded",
      "created_at": 1699185501,
      "updated_at": 1699185501,
      "object": "deployment"
    },
    {
      "scale_settings": {
        "scale_type": "standard"
      },
      "model": "text-embedding-ada-002",
      "owner": "organization-owner",
      "id": "EMBEDDING", ,  // <=== engine: deployment Name
      "status": "succeeded",
      "created_at": 1700045037,
      "updated_at": 1700045215,
      "object": "deployment"
    }
  ],
  "object": "list"
}
```

#### Referenties
  * Wat zijn omgevings variabelen.   ====> https://geekflare.com/nl/python-environment-variables/
  * Dotenv installeer pagina.        ====> https://github.com/theskumar/python-dotenv

### Stap 4

De UnstructuredFileLoader ondersteunt het laden van vele bestandstypes zoals PDF's, PPT's, afbeeldingen, enz.
In het onderstaande voorbeeld wordt een PDF bestand genaamd "Sample.pdf" geladen in de variabele "documents".
Dit bestand moet in dezelfde directory staan als het notebook.

````python
'''
# ====> from langchain.document_loaders import UnstructuredFileLoader
# laad een document "sample.pdf' in de variabele "documents"
# Partitioning Strategy: "fast" or "accurate"
# Partitioning Mode: "single", "elements", or "paged
# single = all the text from all elements are combined into one (default)
# elements = maintain individual elements
# paged = texts from each page are only combined
'''

loader = UnstructuredFileLoader('Sample.pdf', strategy='fast')
documents = loader.load()

# toon de inhoud van het "sample.pdf" document
display(documents)
  ````
Referenties
* Uitleg langchain.document_loaders. =====> https://python.langchain.com/docs/integrations/document_loaders/unstructured_file


### Stap 5 (niet nodig voor kleine bestanden)

De CharacterTextSplitte functie kan stukken tekst in kleine stukken verdelen. Er wordt gesplitst per karakter/leesteken  (standaard met: "\n\n") en de lengte van de "chunck"  wordt bepaald op basis van het aantal leestekens (in het Engels: Characters).


````python
'''
===> from langchain.text_splitter import CharacterTextSplitter
'''
text_splitter = CharacterTextSplitter(chunk_size=8000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

display(texts)
  ````
Referenties
* Uitleg langchain.document_loaders. =====> https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/character_text_splitter


### Stap 6

Emmbedding is een techniek om tekstuele data om te zetten in een numerieke representatie. Deze numerieke representatie kan vervolgens worden gebruikt om
via een OpenAI taal model zoals gpt-35-turbo een vraag te stellen. De onderstaande code toont hoe je een AzureOpenAIEmbeddings object kunt maken zodat je daarna een vraag kunt stellen over het via Langchain ingelezen PDF document.


```python
'''
====> from langchain.embeddings import AzureOpenAIEmbeddings
====> from langchain.vectorstores import Chroma
====> from langchain.chains import RetrievalQA
====> from langchain.llms import AzureOpenAI
'''
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="EMBEDDING",  #<==== set to the deployment name with  "text-embedding-ada-002 (Version 2)"  model
    openai_api_version = "2023-09-15-preview",
)

display(embeddings)
doc_search = Chroma.from_documents(texts,embeddings)

# ===> engine: deployment Name with "gpt-35-turbo" model <===

chain = RetrievalQA.from_chain_type(llm=AzureOpenAI(model_kwargs={'engine':'DAVINCI'}), 
chain_type='stuff', retriever = doc_search.as_retriever())
```

  ##### Referenties

  * https://python.langchain.com/docs/integrations/text_embedding/azureopenai

  ### Stap 7

  Stel je vraag.


  ````python
query = 'What is the main topic of the text? Use only one sentence of max 20 words'
chain.run(query)
 ````


 <br>

***********
# demo

Een voorbeeld van een werkende RAG implementatie met Azure + LangChain + OpenAI is te in te zien <br> via de volgende:
[Google Colab Notebook:](https://colab.research.google.com/drive/1bNbHBXzP1tnDU-xNDRLLeCLFkK3XnI7K#scrollTo=mcw9fPYBOh7b)
genaamd : LangChain-GPT35_v01.ipynb
<br>
<br>

<img align="center" width="1000" height="400" src=".\screenshot_colab.png">

<br>