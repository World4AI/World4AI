<svelte:head>
    <title>World4AI | Programming | Python Setup</title>
    <meta name="description" content="There are several ways to use Python. For absolute beginners cloud programming is a great choice. Over time beginners should try out different setups and figure out what suits them best.">
</svelte:head>

<h1>Python setup</h1>

<p><em>How can you setup Python?</em></p>
<div class="separator"></div>

<p>There are probably a dozen ways to run Python. The setup process depends on whether you want to run Python locally or in the cloud. For a beginner it makes sense to create an online account with one of the cloud services. The process is usually hassle-free and the likelihood of giving up because of the setup process is relatively low compared to the local installation. Intermediate and advanced users on the other hand should be able to install and manage Python on many different systems. Especially when you intend to deploy your machine learning models on the web it is essential to know how to setup Python and the required packages on a Linux distribution.</p>

<div class="separator"></div>

<h2>Python in the cloud</h2>

<p>If you start out and would like to jump into coding, a cloud provider is a great choice.</p>
<ul>
    <li><a target="_blank" href="https://replit.com/">replit</a> is free to use, but requires registration.</li>
    <li><a target="_blank" href="https://www.online-python.com/">ONLINE PYTHON</a> is still in beta, but does not require any form of registration.</li> 
    <li><a target="_blank" href="https://colab.research.google.com/">Google Colab</a> is the usual choice when you need GPU computing (as is often the case with deep lerning) but you are on a budget. Keep in mind, that you need a google account to use the service. Basic usage is free, but advanced tiers are available for paying customers.</li> 
</ul>

<p>If you don't know where to start, choose one of the above options. You can switch to a local setup, once you have mastered the basics of Python.</p>
    
<div class="separator"></div>

<h2>Local Setup</h2>

<p>Installing Python on a local machine is slightly more complex, but if you are serious about becoming a practitioner or researcher you have to learn how to install and use Python locally.</p>
<ul>
    <li><a target="_blank" href="https://www.anaconda.com/">Anaconda </a>is probably the easiest way to get started with Python locally. It comes preinstalled with many machine learning packages and usually works out of the box.</li>
    <li><a target="_blank" href="https://docs.conda.io/en/latest/miniconda.html">Miniconda </a> is the little brother of Anaconda. Miniconda comes with almost no packages and gives the user much more freedom to pick and choose. Unlike Anaconda, Miniconda does not have a graphical interface (GUI) and all installations have to be done from the terminal. This is the option for more experienced users, but you can transition from Anaconda to Miniconda rather smoothly, as the same terminal commands are also available in Anaconda. Think about it the following way: Anaconda = Conda + Packages + GUI.</li>
</ul>
<h2>Text Editor</h2>
<p>If you don't use a cloud provider for your Python programming, a text editor (or an integrated development environment - IDE). A good text editor provides features like syntax highlighting, quick navigation through the project, autocompletion and much more. There are probably hundreds of text editors. Below is a selection to get you started.</p>
<ul>
    <li><a target="_blank" href="https://thonny.org">Thonny</a> is an IDE that is specifically designed for beginners. The IDE comes packaged with Python, therefore you don't need to separately install Python.</li>
    <li><a target="_blank" href="https://jupyter.org">Jupyter Lab</a> is not a typical text editor or IDE. When you start Jupyter Lab, a local server is initiated and you can start programming in your browser. The whole idea of Jupyter is hard to convey using just words, but it comes preinstalled with Anaconda, so try it out if you are curious. Google Colab is also an online implementation of Jupyter.</li>
    <li><a target="_blank" href="https://code.visualstudio.com">Visual Studio Code</a> is developed by Microsoft, but is open sourced. It has become probably the most recommended code editor, especially among web developers. It offers a lot of useful features and is easy to setup, but there is some controvery surrounding the editor. The editor collects telemetry out of the box, including user data. For that reason there is a "clean" version of VSCode, called <a target="_blank" href="https://vscodium.com">VSCodium</a>. This is basically the same editor, but free of spyware.</li>
    <li><a target="_blank" href="https://www.vim.org/">Vim</a> and <a target="_blank" href="https://neovim.io">NeoVim</a> are text editors, which roots date back many, many decades. But do not let the old age fool you. Anything you can imagine doing with an editor is possible with Vim. It is especially powerful when it comes to the speed with which you can edit text. In Vim you do not use the mouse for text editing, as mouse interaction is slow. The learning curve is steep, but in our opinion the advantages outweigh the initial time investment. NeoVim is a fork (somewhat of a copy) of Vim. NeoVim has become extremely popular over the last decades, due to better initial configuration, Lua support and much more. This editor is not for beginners, but keep Vim in mind. Learning Vim is worthwhile.</li>
</ul>

<div class="separator"></div>
<style>
    a {
        color: var(--text-color);
    }

    li {
        padding-bottom: 10px;
    }
</style>
