<script>
  const navigation = [
    { name: "Home", link: "/" },
    {
      name: "Blocks",
      link: "/blocks/introduction",
      children: [
        { name: "Introduction", link: "/blocks/introduction" },
        { name: "Deep Learning", link: "/blocks/deep_learning/introduction" },
      ],
    },
    { name: "Donate", link: "/donate" },
    { name: "About", link: "/about" },
  ];

  let isHamburgerClicked = false;
  function hamburgerEventHandler() {
    isHamburgerClicked = !isHamburgerClicked;
  }
</script>

<header
  class="fixed top-0 left-0 w-full z-50 select-none py-2 px-4 bg-white"
>
  <div class="flex flex-row justify-between items-center container mx-auto">
    <div class="flex items-center space-x-3">
      <a href="/">
        <img src="/logo/logo.svg" class="max-w-[20px]" alt="World4AI Logo" />
      </a>
      <span class="font-bold tracking-[5px] uppercase">World4AI</span>
    </div>
    <!--Hamburger Menu-->
    <button
        class="md:hidden"
        class:active={isHamburgerClicked}
        on:click={hamburgerEventHandler}>
        <span
          id="top"
          class="pointer-events-none block w-5 h-[1px] mb-1.5 bg-black"></span>
        <span
          id="mid"
          class="pointer-events-none block w-5 h-[1px] mb-1.5 bg-black"></span>
        <span id="bot" class="pointer-events-none block w-5 h-[1px] bg-black"
        ></span>
    </button>

    <!-- navigation menu -->
    <nav
      class="bg-white hidden md:block drop-shadow-md md:drop-shadow-none absolute md:relative right-2 top-full p-2 md:p-0"
      class:hidden={!isHamburgerClicked}
    >
      <ul
        class="flex flex-col md:flex-row md:space-x-10 lg:space-x-20 font-bold"
      >
      {#each navigation as nav}
        <li class="group relative">
          <a class="inline-block p-2 hover:bg-gray-100" href={nav.link}>
            {nav.name}
          </a>
          {#if nav.children}
            <ul class="z-10 bg-white text-left whitespace-nowrap md:hidden md:group-hover:block md:absolute md:drop-shadow-lg">
            {#each nav.children as child}
              <li>
                <a
                  class="inline-block py-2 px-4 w-full hover:bg-gray-100"
                  href={child.link}
                >
                  {child.name}
                </a>
              </li>
            {/each}
            </ul>
          {/if}
        </li>
      {/each}
      </ul>
    </nav>
  </div>
</header>