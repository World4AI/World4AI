<script>
  import Question from '$lib/Question.svelte';
</script>

<svelte:head>
    <title>World4AI | Programming | Python Setup</title>
    <meta name="description" content="There are several ways to use Python. For absolute beginners cloud programming is a great choice. Over time beginners should try out different setups and figure out what suits them best.">
</svelte:head>

<h1>Python setup</h1>
<Question>How can you start writing Python programs?</Question>
<div class="separator"></div>

<p>Generally speaking you only need two things to write and run a Python program: an installation of <strong>Python</strong> on the system and a <strong>text editor</strong>.</p>
<div class="flex-center">
  <svg width="500" height="800" version="1.1" viewBox="0 0 500 800" xmlns="http://www.w3.org/2000/svg">
   <g fill="none" stroke="var(--text-color)">
    <rect x="20" y="20" width="460" height="760" ry="10" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/>
    <path d="m100 20v760" stroke-width="1px"/>
    <g fill="var(--text-color)" stroke-linecap="round" stroke-linejoin="round">
     <g stroke-width="2">
      <rect x="30" y="50" width="10" height="10" ry="0"/>
      <rect x="30" y="110" width="10" height="10" ry="0"/>
      <rect x="30" y="135" width="10" height="10" ry="0"/>
     </g>
     <rect x="50" y="65" width="35" height="10" ry="0" stroke-width="2"/>
     <rect x="50" y="80" width="35" height="10" ry="0" stroke-width="2"/>
    </g>
    <g stroke-width="1px">
     <path d="m35 60v10h15"/>
     <path d="m35 75v10h15"/>
     <path d="m100 720h380"/>
     <path d="m110 730 10 10-10 10"/>
     <path d="m115 730 10 10-10 10"/>
     <path d="m120 730 10 10-10 10"/>
     <path d="m30 55h10"/>
     <path d="m35 110v10"/>
     <path d="m30 115h10"/>
     <path d="m35 135v10"/>
     <path d="m30 140h10"/>
    </g>
    <g fill="var(--text-color)" opacity=0.9 stroke-linecap="round" stroke-linejoin="round">
     <rect x="112.5" y="41.25" width="187.5" height="18.75" ry="9.375" stroke-width="2"/>
     <g stroke-width="2">
      <rect x="150" y="78.75" width="309.38" height="18.75" ry="9.375"/>
      <rect x="150" y="116.25" width="309.38" height="18.75" ry="9.375"/>
      <rect x="150" y="153.75" width="309.38" height="18.75" ry="9.375"/>
     </g>
     <rect x="112.5" y="247.5" width="187.5" height="18.75" ry="9.375" stroke-width="2"/>
     <rect x="150" y="285" width="309.38" height="18.75" ry="9.375" stroke-width="2"/>
     <rect x="150" y="322.5" width="309.38" height="18.75" ry="9.375" stroke-width="2"/>
     <rect x="112.5" y="407.25" width="187.5" height="18.75" ry="9.375" stroke-width="2"/>
     <g stroke-width="2">
      <rect x="150" y="444.75" width="309.38" height="18.75" ry="9.375"/>
      <rect x="150" y="482.25" width="309.38" height="18.75" ry="9.375"/>
      <rect x="150" y="517.25" width="309.38" height="18.75" ry="9.375"/>
      <rect x="150" y="552.25" width="309.38" height="18.75" ry="9.375"/>
     </g>
    </g>
    <g stroke-width="1px">
     <path d="m20 390h80"/>
     <path d="m30 730h60"/>
     <path d="m30 720h60"/>
     <path d="m30 710h60"/>
    </g>
   </g>
  </svg>
</div>
<p>The setup process depends on whether you want to run Python locally or in the cloud. For a beginner it is not the worst idea to create an online account with one of the cloud service providers. The process is usually hassle-free and the likelihood of giving up because of the setup process is relatively low compared to the local installation. Intermediate and advanced users on the other hand should be able to install and manage Python on many different systems.</p>
<p>In this section we are going to provide some recommendations for a simple setup, that beginners can implement without a lot of resistance. Additionally we are going to leave some recommendations of things to look at, once the beginner feels more comfortable with Python and wants to progress.</p>
<div class="separator"></div>

<h2>Setup for beginners</h2>
<h3>Python in the cloud</h3>
<ul>
    <li><a target="_blank" href="https://replit.com/">Replit</a> gives you access to a text editor and and different programming languages (including Python) in the cloud. The starter package is free and should be more than sufficient to get you started.</li>
    <li><a target="_blank" href="https://colab.research.google.com/">Google Colab</a> is the usual choice when you need GPU computing (as is often the case with deep lerning) but you are on a budget. Keep in mind, that you need a google account to use the service. Basic usage is free and advanced tiers are available for paying customers.</li>
</ul>

<h3>Local Setup</h3>
<ul>    
  <li><a target="_blank" href="https://thonny.org">Thonny</a> is an IDE that is specifically designed for beginners. The IDE comes packaged with Python, therefore you don't need to separately install Python. This is most likely the easiest way to get started locally.</li>
</ul>
<div class="separator"></div>

<h2>Advanced Setup</h2>
<ul>
    <li><a target="_blank" href="https://www.anaconda.com/">Anaconda </a>is an easy way to get started with Python locally. It comes preinstalled with many machine learning packages and usually works out of the box.</li>
    <li><a target="_blank" href="https://docs.conda.io/en/latest/miniconda.html">Miniconda </a> is the little brother of Anaconda. Miniconda comes with almost no packages and gives the user much more freedom to pick and choose. Unlike Anaconda, Miniconda does not have a graphical interface (GUI) and all installations have to be done from the terminal. This is the option for more experienced users, but you can transition from Anaconda to Miniconda rather smoothly, as the same terminal commands are also available in Anaconda. Think about it the following way: Anaconda = Conda + Packages + GUI.</li>
    <li><a target="_blank" href="https://jupyter.org">Jupyter Lab</a> is not a typical text editor or IDE. When you start Jupyter Lab, a local server is initiated and you can start programming in your browser. The whole idea of Jupyter is hard to convey using just words, but it comes preinstalled with Anaconda, so try it out if you are curious. Google Colab is also an online implementation of Jupyter.</li>
    <li><a target="_blank" href="https://code.visualstudio.com">Visual Studio Code</a> is developed by Microsoft, but is open sourced. It has become probably the most recommended code editor, especially among web developers. It offers a lot of useful features and is easy to setup, but there is some controvery surrounding the editor. The editor collects telemetry out of the box, including user data. For that reason there is a "clean" version of VSCode, called <a target="_blank" href="https://vscodium.com">VSCodium</a>. This is basically the same editor, but free of spyware.</li>
    <li><a target="_blank" href="https://www.vim.org/">Vim</a> and <a target="_blank" href="https://neovim.io">NeoVim</a> are text editors, which roots date back many, many decades. But do not let the old age fool you. Anything you can imagine doing with an editor is possible with Vim. It is especially powerful when it comes to the speed with which you can edit text. In Vim you do not use the mouse for text editing, as mouse interaction is slow. The learning curve is steep, but in our opinion the advantages outweigh the initial time investment. NeoVim is a fork (somewhat of a copy) of Vim. NeoVim has become extremely popular over the last decades, due to better initial configuration, Lua support and much more. This editor is not for beginners, but keep Vim in mind. Learning Vim is worthwhile.</li>
</ul>

<div class="separator"></div>
<style>
    svg {
      height: 50vh;
    }
    a {
        color: var(--text-color);
    }

    li {
        padding-bottom: 10px;
    }
</style>
