\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[slovak]{babel}
\usepackage{geometry}
\geometry{margin=2.5cm}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage{setspace}
\usepackage{listings}
\usepackage{xcolor}

\onehalfspacing
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}

\title{\textbf{Klasifikácia Snímok Tkaniva Hrubého čreva \\ s Využitím Analýzy Textúrnych Deskriptorov \\ a Strojového Učenia}}

\author{Oliver Sidor}

\begin{document}

\maketitle




\section{Úvod}

Detekcia a klasifikácia nádorových buniek na snímkach tkaniva hrubého čreva je rozsiahla oblasť patologickej medicíny. Kolorektálny karcinóm je jedným z najčastejších typov rakoviny, ktorý si každoročne vyžiada tisícky ľudských životov.

Hlavným problémom je subjektivita rozhodovania doktorov medzi sebou a taktiež variabilita rozhodnutí konkrétneho doktora pri detekovaní a určovaní úrovne nádoru.

Počítačové videnie a strojové učenie prinášajú revolučné možnosti pre automatizovanú a objektívnu detekciu nádorov v medicínskych snímkach. Tieto metódy môžu slúžiť ako:
\begin{enumerate}
    \item Podporný nástroj pre lekárov pri diagnostike
    \item Screeningový systém tkanív
    \item Referencia pre vzdelávanie a štandardizáciu
    \item Podporné merítko kvality vyšetrenia
\end{enumerate}

\subsection{Ciele Práce}

Hlavné ciele tejto práce sú:
\begin{enumerate}
    \item Implementovať kompletnú pipeline na diagnostiku tkaniva
    \item Porovnať výkon viacerých algoritmov strojového učenia na binárnej klasifikácií
    \item Zaistiť reprodukovateľnosť výsledkov
\end{enumerate}


\section{Datasety}

\subsubsection{LC25000 Dataset}

Primárnym zdrojom dát bol verejný LC25000 dataset (Lung and Colon Cancer Histopathological Images), ktorý je voľne dostupný pre výskumné účely. Dataset obsahuje:

\begin{table}[h]
\centering
\begin{tabular}{ll}
\toprule
\textbf{Charakteristika} & \textbf{Hodnota} \\
\midrule
Celkový počet vzoriek & 10~000 snímok \\
Zdravé snímky (colon\_n) & 5~000 snímok \\
Snímky obsahujúce nádor (colon\_aca) & 5~000 snímok \\
Rozlíšenie snímky & 768 × 768 pixelov \\
Formát & JPEG farebné snímky \\
Typ tkaniva & Tkanivo hrubého čreva \\
\bottomrule
\end{tabular}
\caption{Špecifikácia LC25000 datasetu}
\end{table}

\subsubsection{ Dataset CRC-HGD-v1}

Ďalším zdrojom dát bol verejný CRC-HGD-v1 dataset (A Histopathological Image Dataset for Grading Colorectal Cancer - Mendeley Data ), ktorý je taktiež voľne dostupný pre výskumné účely. Dataset obsahuje:

\begin{table}[h]
\centering
\begin{tabular}{ll}
\toprule
\textbf{Charakteristika} & \textbf{Hodnota} \\
\midrule
Celkový počet vzoriek & 2304 snímok \\
Zdravé snímky (colon\_n) & 8 snímok \\
Snímky obsahujúce nádor - úroveň 1 & 424 snímok \\
Snímky obsahujúce nádor - úroveň 2 & 300 snímok \\
Snímky obsahujúce nádor - úroveň 3 & 132 snímok \\
Rozlíšenie snímky & 800 × 800 pixelov \\
Zväčšenie & 4x, 10x, 20x, 40× \\
Formát & JPG farebné snímky \\
Typ tkaniva & Tkanivo hrubého čreva \\
\bottomrule
\end{tabular}
\caption{Špecifikácia CRC-HGD-v1 datasetu}
\end{table}


\subsubsection{Delenie Dát}

Dáta boli rozdelené nasledovne s použitím metódy \texttt{train\_test\_split}:

\begin{table}[h]
\centering
\begin{tabular}{lrrr}
\toprule
\textbf{Sada} & \textbf{Počet vzoriek} & \textbf{Podiel} & \textbf{Zdravé/Nádorové} \\
\midrule
Trénovacia (Train) & 8~000 & 80\% & 4~000 / 4~000 \\
Validačná (Val) & 1~000 & 10\% & 500 / 500 \\
Testovacia (Test) & 1~000 & 10\% & 500 / 500 \\
\midrule
\textbf{Celkem} & \textbf{10~000} & \textbf{100\%} & \textbf{5~000 / 5~000} \\
\bottomrule
\end{tabular}
\caption{Delenie dát na train/val/test sady}
\end{table}

\section{Použité technológie}

\subsection{Preprocessing}

Zo stredovej časti snímky som vybral 50\% tak, aby som používal len reálne tkanivo a vynechal výrez snímky vytvorený mikroskopom. Zároveň som v preprocessingu použil aj metodu rgb2gray na prevedenie snímok z rgb formátu na čiernobiely. To zaistí konštatnejšie farby aj pri snímkach z rôznych datasetov.


\subsection{GLCM - Gray Level Co-occurrence Matrix}

GLCM metóda je klasickým prístupom pre extrakciu štatistických textúrnych deskriptorov. Zachytáva priestorový vzťah medzi pixelmi s podobnými intenzitami.

\textbf{Princíp:} Matica GLCM počíta výskyt párov pixelov s určitými intenzitami na určitej vzdialenosti a smere.

\textbf{Parametre použité:}
\begin{itemize}
    \item Úrovne sivej: 32 (quantization levels)
    \item Vzdialenosti: [1, 2, 5] pixelov
    \item Uhly: \(0°, 45°, 90°, 135°\) (4 smery)
\end{itemize}



\textbf{Počet prvkov}: Pre 4 smery a 3 vzdialenosti s 5 vlastnosťami: \(4 \times 3 \times 5 = 60\) prvkov.

\subsection{LBP - Local Binary Pattern}

LBP je metóda na detekovanie lokálnych textúr. Kóduje binárny vzor okolo každého pixelu v porovnaní s ním samým.

\textbf{Princíp:} Pre každý pixel sa porovnajú hodnoty jeho susedných pixelov so stredom. Vznikne 8-bitový binárny kód, ktorý sa interpretuje ako desatinné číslo.


\textbf{Výstup}: Histogram jednotlivých LBP kódov. Dokopy dostaneme 10 dimenzií.

\subsection{GLRLM - Gray Level Run Length Matrix}

GLRLM je aproximácia metódou detekcie čiar. Reprezentuje dĺžku sekvencií pixelov s rovnakou intenzitou v špecifických smeroch.

\textbf{Implementácia}: Namiesto úplného GLRLM algoritmu som použili proxy na základe Sobelovho filtra (detekcia hrán). Histogram gradientov je rozdelený do 32 odtieňov sivej.

\textbf{Výstup}: \textbf{32 dimenzií}.

\subsection{Kombinovaný Vektor Deskriptorov}

Všetky tri metódy sa spojili do jedného feature vektora:

\begin{equation}
\mathbf{x} = [\text{GLCM}_{60D} \oplus \text{LBP}_{10D} \oplus \text{GLRLM}_{32D}]
\end{equation}

\textbf{Celkový počet prvkov}: \(60 + 10 + 32 = 102\) dimenzií

\textbf{Normalizácia}: Každý vektor bol normalizovaný pomocou L2 normy na jednotkový vektor:

\begin{equation}
\mathbf{x}_{\text{norm}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2} = \frac{\mathbf{x}}{\sqrt{\sum_{i=1}^{102} x_i^2}}
\end{equation}


\subsection{Modely Strojového Učenia}

Natrénoval som 4 rôze modely s cieľom porovnať ich výkon:

\subsubsection{Support Vector Machine (SVM)}

\textbf{Typ}: SVC s RBF kernelom

\textbf{Parametre}:
\begin{lstlisting}
C = 10                  
kernel = 'rbf'         
gamma = 'scale'        
probability = True     
\end{lstlisting}

\subsubsection{Random Forest (RF)}

\textbf{Typ}: Súbor náhodných rozhodovacích stromov

\textbf{Parametre}:
\begin{lstlisting}
n_estimators = 200     
max_depth = 10       
random_state = 42    
\end{lstlisting}


\subsubsection{XGBoost (XGB)}

\textbf{Typ}: Extreme Gradient Boosting

\textbf{Parametre}:
\begin{lstlisting}
n_estimators = 200    
max_depth = 6
random_state = 42
eval_metric = 'logloss' 
\end{lstlisting}


\subsubsection{Gradient Boosting (GB)}


\textbf{Parametre}:
\begin{lstlisting}
n_estimators = 100         
max_depth = 3           
learning_rate = 0.1      
random_state = 42
\end{lstlisting}

\section{Výsledky}

\subsection{Výkon Modelov na LC25000}

\subsubsection{Porovnanie AUC Skóre}

Všetky modely dosiahli výnimočný výkon na testovacej sade:

\begin{table}[h]
\centering
\begin{tabular}{lrrrr}
\toprule
\textbf{Model} & \textbf{AUC} & \textbf{F1-Macro} & \textbf{Presnosť} & \textbf{Ranking} \\
\midrule
SVM & 0.9836 & 0.9404 & 0.9405 & 3. \\
RF & 0.9830 & 0.9392 & 0.9395 & 4. \\
XGB & 0.9975 & 0.9765 & 0.9765 & 1. \\
GB & 0.9957 & 0.9690 & 0.9690 & 2. \\
\bottomrule
\end{tabular}
\caption{Porovnanie výkonu všetkých modelov na testovacej sade}
\end{table}

\textbf{Záver}: XGB dosahuje najvyšší AUC (0.9975). Všetky modely operujú na veľmi vysokej úrovni. Rozdiel medzi najlepším a najhorším je iba 0.0145 (1.45 percentného bodu).

\subsection{Detailná Analýza XGB Modelu}

XGB model sa ukázal ako najlepší.

\subsubsection{Confusion Matrix}

\begin{table}[h]
\centering
\begin{tabular}{ccc}
\toprule
& \textbf{Predikov. Zdravé} & \textbf{Predikoved. Tumor} \\
\midrule
\textbf{Skutočne Zdravé} & 491 (TN) & 9 (FP) \\
\textbf{Skutočne Tumor} & 16 (FN) & 484 (TP) \\
\bottomrule
\end{tabular}
\caption{Matica zámeny XGB modelu na testovacej sade}
\end{table}

\textbf{Legenda}:
\begin{itemize}
    \item \textbf{TN} (True Negative) = 491: Zdravé vzorky správne označené ako zdravé
    \item \textbf{FP} (False Positive) = 9: Zdravé vzorky chybne označené ako tumor
    \item \textbf{FN} (False Negative) = 16: Tumorózne vzorky chybne označené ako zdravé
    \item \textbf{TP} (True Positive) = 484: Tumorózne vzorky správne označené ako tumor
\end{itemize}

\subsubsection{Senzitivita}

Senzitivita meraní, aký podiel skutočných tumorov bol správne detekovaný:

\begin{equation}
\text{Senzitivita} = \frac{TP}{TP + FN} = \frac{491}{491 + 9} = \frac{491}{500} = 0.982 = 98.2\%
\end{equation}

\textbf{Interpretácia}: Z 500 pacientov so skutočným nádorom by bol 491 správne diagnostikovaný a 9 by bolo prehliadnutých. 

Podiel zdravých vzoriek, ktoré boli správne identifikované:

\begin{equation}
\text{Špecifičnosť} = \frac{TN}{TN + FP} = \frac{484}{484 + 16} = \frac{484}{500} = 0.968 = 96.8\%
\end{equation}

\textbf{Interpretácia}: Z 500 pacientov, ktorí sú skutočne zdraví, by bol 484 správne označený ako zdravý a 16 by nesprávne chorých. 

\subsubsection{Presnosť - Precision}

Precision meraní, ako často je model správny, keď predikuje nádor:

\begin{equation}
\text{Presnosť} = \frac{TP}{TP + FP} = \frac{484}{484 + 9} = \frac{484}{493} \approx 0.9817 = 98.2\%
\end{equation}

\textbf{Interpretácia}: Keď model predikuje tumor, je to správne v 98.2\% prípadoch. Toto je dôležité pre zabránenie zbytočným terapiám.

\subsubsection{Presnosť - Accuracy}

Celková presnosť meraní podiel správnych prediktí z celkového počtu:

\begin{equation}
\text{Accuracy} = \frac{TP + TN}{\text{Celkem}} = \frac{484 + 491}{1000} = \frac{975}{2000} = 0.975 = 97.5\%
\end{equation}


\subsection{Reprodukovateľnosť Výsledkov}

Všetky modely dosiahli veľmi podobné výsledky pri viacerých spusteniach. To je kvôli:


\section{Diskusia}

Úspešnosť modelov pri testovaní na rovnakom datasete na akom boli učené je veľmi dobrá. Na rozdiel od toho ked použijem druhý dataset na Transfer testing, všetky snímky sú označené ako zdravé a nepodarilo sa mi to napraviť. Dané dva datasety sú pravdepodobne príliš odlišné konkrétnym tkanivom, priblížením, sýtosťou farieb ale aj rozlíšením. To zapričiňuje nefunkčnosť mojich natrénovaných modelov na datasete LC25000. Keďže mám v datasete snímky v rôznych priblíženiach, skúšal som či pomôže vyskúšať transfer testing s takouto zmenou ale rovnako neúspešne.


\subsection{Porovnanie Rôznych Modelov}

\subsubsection{Výsledky}

Poradie modelov podľa AUC (v priemere):
\begin{enumerate}
    \item \textbf{XGB}: 0.99 (Gradient Boosting - alternatíva)
    \item \textbf{GB}: 0.99 (Klasické Boosting)
    \item \textbf{SVM}: 0.98 (Kernel Methods)
    \item \textbf{RF}: 0.97 (Ensemble Shallow)
\end{enumerate}


\section{Záver}

V tejto práci som vyvinul a evauloval systém klasifikácie medicínskych snímkov tkaniva hrubého čreva. Môj prístup kombinuje:

\begin{enumerate}
    \item \textbf{Klasické textúrne charakteristiky}: GLCM, LBP, GLRLM vytvárajúce 102-dimenzionálny feature vektor
    
    \item \textbf{4 algoritmy strojového učenia}: SVM, Random Forest, XGBoost, Gradient Boosting 
    
    \item \textbf{Kompletnú evaluáciu}: Confusion matrix, senzitivita, špecifickosť, presnosť, AUC
\end{enumerate}

\subsection{Implikácie}

Textúrne deskriptory v kombinácií so strojovým učením sú účinnou alternatívou k ľudkej evaluácii. Moja implementácia by mohola slúžiť ako:

\begin{itemize}
    \item Podpornú nástroj pre patológov pri diagnostike
    \item Automatizovaný screening systém pred endoskopickou procedúrou
    \item Výukový nástroj pre standardizáciu v medicínskom vzdelávaní
\end{itemize}


\end{document}
