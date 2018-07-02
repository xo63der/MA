%%
%% This is file `unibrief.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% unibrief.dtx  (with options: `class')
%% 
%% Copyright (C) 1991-1997 by Michael Lenzen. All rights reserved.
%% 
%% 
%% This file is part of the unibrief package
%% 
%% IMPORTANT NOTICE:
%% 
%% You are not allowed to change this file.  You may however copy this file
%% to a file with a different name and then change the copy.
%% 
%% You are NOT ALLOWED to distribute this file alone.  You are NOT ALLOWED
%% to take money for the distribution or use of this file (or a changed
%% version) except for a nominal charge for copying etc.
%% 
%% You are allowed to distribute this file under the condition that it is
%% distributed together with all files mentioned below.
%% 
%% If you receive only some of these files from someone, complain!
%% 
%% Error Reports in case of UNCHANGED versions to
%% 
%%                           Michael Lenzen
%%                           Zur Belsmuehle 15
%%                           53347 Alfter
%%                           Germany
%%                Internet:  <lenzen@vivienne.shw.com>
%%                           <michael@theory.cs.uni-bonn.de>
%% 
\def\filedate{1997/09/15}
\def\fileversion{2.2}
\def\filedate{1997/09/15}
\def\fileversion{2.2}
\def\filename{dbvbrief.cls}
\NeedsTeXFormat{LaTeX2e}[1997/06/01]

\ProvidesClass{dbvbrief}[\filedate\space v\fileversion\space dbvbrief class]
\RequirePackage[utf8]{inputenc}

\newdimen\VorschubH
\newdimen\VorschubV
\VorschubH = 0.0mm
\VorschubV = 2.0mm

%%%%%%%%%%%%%%% Option Part %%%%%%%%%%%%%%%%%%%%
\RequirePackage{ifthen}
\usepackage{times}
\usepackage{graphicx}

\newboolean{colorlogo}
\setboolean{colorlogo}{false}
\DeclareOption{colorlogo}{\setboolean{colorlogo}{true}}

\DeclareOption{10pt}{\def\@schrift{0}}
\DeclareOption{11pt}{\def\@schrift{1}}
\DeclareOption{12pt}{\def\@schrift{2}}
\DeclareOption{german}{
  \def\sprache{german}
  \def\telefaxtext{{Fax\/:}}
  \def\telefontext{{Tel\/:}}
  \def\emailtext{{\footnotesize E--Mail\/:}}
  \def\httptext{{\footnotesize HTTP\/:}}
  \def\betrefftext{{\normalsize Betr.:}}
  \def\datumtext{\footnotesize\textsc{}}
  \def\datum{\number\day.~\ifcase\month\or Januar\or Februar\or
    M\"arz\or April\or Mai\or Juni\or Juli\or August\or September\or
    Oktober\or November\or Dezember\fi \space\number\year}}
\newif\iffenstermarken
\DeclareOption{fenster}{\fenstermarkentrue}
\newif\iffaltmarken
\DeclareOption{falt}{\faltmarkentrue}

\ExecuteOptions{11pt,german} \ProcessOptions
%%%%%%%%%%%%%%% Option Part End %%%%%%%%%%%%%%%%
                                                      


\LoadClass[1\@schrift pt, a4paper, oneside, final]{letter}[1997/04/16]

\normalsize

\setlength\oddsidemargin{0\p@}
\setlength\evensidemargin{\oddsidemargin}
\setlength\marginparwidth{90\p@}
\setlength\marginparsep{11\p@}
\setlength\marginparpush{5\p@}
\setlength\topmargin{-12.69mm}
\setlength\headheight{4.23mm}
\setlength\headsep{6.35mm}
\setlength\topskip{3.38mm}
\addtolength\footskip{4.23mm}

%\setlength\textheight{178.88mm}
\setlength\textheight{171.88mm}
\setlength\textwidth{165mm}

\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}

\renewcommand\baselinestretch{1}

\advance\topmargin by \VorschubV
\advance\oddsidemargin by \VorschubH
\setlength\evensidemargin{\oddsidemargin}

\AtEndOfClass{\pagestyle{regularpage}}

\def\Institute#1{\def\institute{#1}}  \def\institute{Institut f\"r Informatik}
\def\Lfe#1{\def\lfe{#1}}            \def\lfe{Lehrstuhl f\"ur Bioinformatik}
\def\Lfename#1{\def\lfename{#1}}            \def\lfename{}

\def\Name#1{\def\name{#1}} \def\name{}
\def\Strasse#1{\def\strasse{#1}} \def\strasse{}
\def\Ort#1{\def\ort{#1}} \def\ort{}
\def\RetourAdresse#1{\def\retouradresse{#1}} \def\retouradresse{}

\def\Telefax#1{\def\telefax{#1}} \def\telefax{}
\def\Telefon#1{\def\telefon{#1}} \def\telefon{}
\def\EMail#1{\def\email{#1}} \def\email{}
\def\HTTP#1{\def\http{#1}} \def\http{}

\def\Postvermerk#1{\def\postvermerk{#1}} \def\postvermerk{}
\def\Adresse#1{\def\adresse{#1}} \def\adresse{}

\def\Datum#1{\def\datum{#1}} \def\datum{\today}

\def\Betreff#1{\def\betreff{#1}} \def\betreff{}

\def\Anrede#1{\def\anrede{#1}} \def\anrede{}
\def\Gruss#1#2{\def\gruss{#1} \def\grussskip{#2}}
    \def\gruss{} \def\grussskip{}

\def\Unterschrift#1{\def\unterschrift{#1}} \def\unterschrift{}

\def\Anlagen#1{\def\anlagen{#1}} \def\anlagen{}

\long\def\Einrueckung#1{\par\begin{tabular}{@{\hspace{1in}}p{5in}@{}}
    #1\end{tabular}\par}

\newif\ifklassisch\klassischfalse
\def\klassisch{\klassischtrue}

\newenvironment{unibrief}{
  \thispagestyle{firstpage}
  \setlength\unitlength{1mm}

  \c@page\@ne
  \interlinepenalty=200
  \clearpage

  \hfill{}\hfill \makebox[37.0mm][r]{
    \begin{tabular}{@{}r@{}}
      \datumtext\\
      \datum
    \end{tabular}}\par

  \vspace{10mm}

  \ifcase\@ptsize\vspace{0.045pt}\or \vspace{-1.555pt}\or
  \vspace{-2.955pt} \fi

  \ifx \betreff\empty \else \ifklassisch \betrefftext \space \space
  \betreff \else \textbf{\betreff} \fi \mbox{}
  \par \vspace{-1\parskip} \vspace{8mm} \fi

  \ifcase\@ptsize\vspace{0.045pt}\or \vspace{-1.555pt}\or
  \vspace{-2.955pt} \fi

  \ifx \anrede\empty \else \anrede \par \fi\nobreak
  \addtolength\textheight{75mm}
  }
        {
  \renewcommand{\baselinestretch}{1.0}
  \ifcase \@ptsize\relax \normalsize \or \small \or \footnotesize \fi

  \vspace{\grussskip} \par \nobreak \stopbreaks \noindent
  \parbox[t]{3.5in}{\raggedright \ignorespaces {\normalsize \ifx
      \gruss\empty \else \gruss \mbox{} \\[16.92mm] \fi \ifx
      \empty\unterschrift \relax \else \ifklassisch
      \textsl{(\unterschrift)} \else \unterschrift \fi}
    \fi\strut} \ifx \anlagen\empty \else \vspace{4mm} \par Anlagen: \anlagen
  \par \fi \ifx \anlagen\empty
  \vspace{4mm}\par \fi }
 

\def\ps@firstpage{
  \renewcommand{\baselinestretch}{1.0}
  \ifcase \@ptsize\relax \normalsize \or \small \or \footnotesize \fi
  \headheight16pt\headsep80mm
  \def\@oddhead{
    \unitlength1mm
    \begin{picture}(0,0)
      \put(50,-13){\parbox{125mm}{\raggedleft\fontsize{14.4pt}{14.4pt}\selectfont Friedrich-Schiller-Universit\"at Jena}}
      \put(-4,-5){\parbox{50mm}{\includegraphics[width=20mm]{jena-uni-logo}}}
      \put(0,-18){\fontsize{7pt}{7pt}\selectfont seit 1558}
      %%   ^^^ Anfang LMU-logo = anfang ort -strasse !!
%\put(-9,-23){\rule{180mm}{0.5pt}}
\put(-4,-15){\rule{20mm}{0.3mm}}
\put(+73.5,-15){\rule{101.5mm}{0.3mm}}
\put(+93.5,-16.5){\rule{81.5mm}{1.5mm}}
      \iffaltmarken \put(-15,-85.16){\line(1,0){3}}
      \put(-15,-182.16){\line(1,0){3}} \fi
      \put( -9, -35.10){\makebox(85,4)
        {\scriptsize \ifx \retouradresse\empty
          \textrm{
            \renewcommand{\arraystretch}{0.9}
            \begin{tabular}{c}
              \name\ $\cdot$\ \if 0\institute\ $\cdot$\ \fi\lfe \\
              Friedrich-Schiller-Universit\"at Jena $\cdot$\ \strasse
              $\cdot$\ \ort
             \end{tabular}}
            \else \textrm{
              \retouradresse} \fi }}
%      \iftrennlinien \put( -9, -37.10){\line(1,0){85}} \fi
      \put(-1,-42.15){\parbox[t]{3in}{\ifx \postvermerk\empty 
          \else \textbf{\postvermerk} \par \vspace{2mm} \fi \adresse}}
%      \put(115,-32.15){\parbox[t]{60mm}{
      \put(105,-32.15){\parbox[t]{60mm}{
          \begin{tabular}[t]{@{}l@{}}
           \bf \name\\
           \institute\\
            \lfe\\
           \ \\
            \strasse \\
            \ort\\[4mm]
          \end{tabular}\\
          \begin{tabular}{@{}l@{}l@{}}
        \telefontext & \telefon\\
        \telefaxtext & \telefax\\
          \end{tabular}\\
          \begin{tabular}[t]{@{}l@{}}
          \email
          \end{tabular} 
        }}
      \iffenstermarken
      \put(-9,-28.15){\line( 1, 0){1}} \put(-9,-28.15){\line( 0,-1){1}}
      \put(76,-28.15){\line(-1, 0){1}} \put(76,-28.15){\line( 0,-1){1}}
      \put(-9,-71.00){\line( 1, 0){1}} \put(-9,-71.00){\line( 0, 1){1}}
      \put(76,-71.00){\line(-1, 0){1}} \put(76,-71.00){\line( 0, 1){1}}
      \fi
    \end{picture}
    \hfill}
  \def\@oddfoot{}
  \def\@evenhead{\@oddhead} \def\@evenfoot{\@oddfoot}}

\def\ps@regularpage{
\headsep20mm
  \headheight36pt\def\@oddhead{
    \unitlength1mm
    \begin{picture}(0,0)
      \put(115,6){\large Friedrich-Schiller-Universit\"at Jena}
      %%   ^^^ Anfang LMU-logo = anfang ort -strasse !!
%\put(-9,-23){\rule{180mm}{0.5pt}}
\put(+100,4){\rule{74mm}{0.5pt}}
\put(+115,3){\rule{59mm}{3pt}}
      \put(-87,0){\makebox(190,15){\normalsize \ifklassisch \textsl{
            \quad\name,\ \datum\quad  \pagename\ \thepage\quad} \else
          \textsc{\pagename\ \thepage\quad}
          \fi}}
%      \iftrennlinien \put(-9,0){\rule{180mm}{0.5pt}} \fi
    \end{picture}\hfill}
  \def\@oddfoot{\empty} \def\@evenhead{\@oddhead}
  \def\@evenfoot{\@oddfoot}}

\def\istsprache{german}
\ifx \sprache\istsprache
\IfFileExists{babel.sty}
{\RequirePackage[\sprache]{babel}[1997/01/23] }
{\IfFileExists{german.sty}
  {\RequirePackage{german}[1997/05/01]}
  {\ClassError{unibrief}
    {Neither babel nor german.sty installed !!!}
    {Get babel or german.sty !!!}}}
\else
\IfFileExists{babel.sty}
  {\RequirePackage[\sprache]{babel}[1997/01/23]}
  {\ClassError{unibrief}
    {Babel not installed !!!}
    {Get babel package !!!}}
\fi

\endinput
%%
%% End of file `unibrief.cls'.
