
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
| 3. xxx
| 4. xxx
| 5. xxx
| 6. DEMO [[DEMO]](#demo).



Deze GitHub Repository geeft inzicht hoe je met behulp van  Generatieve-AI (Gen-AI) je eigen documenten kunt bevragen.



>Disclaimer: deze tekst is door het gebruik van *"gezond verstand'* tot stand gekomen. <br> Artificiële intelligentie [AI] is gebruikt ter verificatie van de gebruikte bronnen + vertaling van Engelstalige teksten.




  Views since 15 juni 2023: [![HitCount](https://hits.dwyl.com/robvdw/HR-DATA-FABRIC/Leer-je-eigen-documenten-bevragen-met-generatieve-AI.svg?style=flat-square)](http://hits.dwyl.com/robvdw/HR-DATA-FABRIC/Leer-je-eigen-documenten-bevragen-met-generatieve-AI)
  <br>
  Unique visitors since 15 juni 2023: [![HitCount](https://hits.dwyl.com/robvdw/HR-DATA-FABRIC/Leer-je-eigen-documenten-bevragen-met-generatieve-AI.svg?style=flat-square&show=unique)](http://hits.dwyl.com/robvdw/HR-DATA-FABRIC/Leer-je-eigen-documenten-bevragen-met-generatieve-AI)

*****
### Dit is een data product gemaakt door <br> het [PROMETHEUS DATA SCIENCE LAB](https://github.com/HR-DATA-FABRIC/PROMETHEUS) <br> van de Hogeschool Rotterdam.

*****







# intro
***********
## Waarom is RAG nodig als ik met LLMs zoals ChatGPT, Co-Pilot & Bard al mijn eigen documenten kan bevragen?

Gen-AI  applicaties gebaseerd op grote taalmodellen (LLMs) verwijst naar elke vorm van machinaal-lerende (ML) kunstmatige intelligentie (AI) die gebruikt maakt van natuurlijke taal verwerkende (NLP) algoritmen. 

Eindgebruikers  kunnen  online gebruik  maken van Gen-AI data-producten via webbrowser userinterfaces.  Hierdoor is het mogelijk om met natuurlijke taal instructies (zogenaamde prompts), binnen enkele seconden, de gewenste content te genereren, in de vorm van  tekst, broncode afbeeldingen, video's, muziek etc.

LLMs worden tijdens hun traingingsfase gevoed met publiekelijk beschikbare content, in de vorm van boeken, video's, audio opnames, databases, artikelen, websites, en open source-broncode. Zo leren LLMs output te creëren die sterk lijkt op door mensen gemaakte, authentieke content. Op het moment dat eindgebruikers gebruik maken van op LLM-gebaseerde gen-AI userinterfaces dan is het onderliggende taal model bevroren en kan niet meer worden getraind.  De Engelse term “pre-trained” wordt veelal gebruikt om dit aan te geven. Het onderliggende taalmodel van de ChatGPT interface is GPT-4 hetgeen staat voor: 4de generatie “Generative Pre-trained Transformer”.

Een groot nadeel van de huidige generatie LLMs is dat ze feitelijk functioneren als algemene (lees general-purpose) kennisbanken, waarvan de hoeveelheid beschikbare gegevens niet uitbreidbaar (lees, bevroren) en niet op één enkele locatie in het model is opgeslagen, maar over verschillende locaties is verspreid (lees, gedistribueerd).
Met andere woorden, de in LLMs opgeslagen kennis is voor eindgebruikers zowel onveranderbaar als ook  onherkenbaar waardoor taalmodellen niet eenvoudig doelgericht zijn te bevragen over een specifiek en actueel kennisdomein. Het is dus  niet mogelijk om te bepalen welke informatie het LLM  benut om antwoorden te genereren, waardoor waarheidsvinding (fact-checking) moeilijk uitvoerbaar is.

Deze tekortkomingen kunnen worden tegengegaan door gebruik te maken van Retrieval-Augmented Generation (RAG). Deze op natuurlijke taal generatie (NLP) gebaseerde AI-technologie kun je opvatten als een onderzoeks- en schrijversduo. Stel je voor dat je een journalist bent die verslag doet van  een natuurramp. Je doet dan eerst onderzoek naar de gebeurtenis, verzamelt relevante artikelen of weer rapporten en gebruikt deze informatie om dit specifieke nieuwsverhaal te schrijven.

RAG doet iets soortgelijks maar dan voor grote-taalmodellen. De retriever-component representeert de journalist die relevante informatie verzamelt, en de generator-component is de schrijver die deze informatie gebruikt om een voor mensen begrijpelijke en waardevolle nieuwsverhaal te schrijven.

***********


# demo
***********
## RAG implementatie met Azure + LangChain + OpenAI