<script>
  import { Github, Twitter, MoreVertical } from "lucide-svelte";
  const navigation = [
    { name: "Home", link: "/" },
    {
      name: "Blocks",
      link: "/blocks/introduction",
      children: [
        { name: "Introduction", link: "/blocks/introduction" },
        {
          name: "Deep Learning",
          link: "/blocks/deep_learning/fundamentals/introduction",
        },
        {
          name: "Reinforcement Learning",
          link: "/blocks/reinforcement_learning/intuition/introduction",
        },
      ],
    },
    { name: "Sponsor", link: "/sponsor" },
    { name: "About", link: "/about" },
  ];

  let isHamburgerClicked = false;
  function hamburgerEventHandler() {
    isHamburgerClicked = !isHamburgerClicked;
  }
</script>

<header
  class="sticky top-0 left-0 w-full z-50 select-none py-2 bg-white border-b"
>
  <div class="container mx-auto flex flex-row justify-between items-center">
    <div class="flex items-center space-x-3">
      <a href="/">
        <img
          src="/logo/logo.svg"
          class="w-[20px] h-[20px] max-w-[20px]"
          alt="World4AI Logo"
        />
      </a>
      <span class="font-bold tracking-[5px] uppercase">World4AI</span>
    </div>
    <!--Hamburger Menu-->
    <button
      class="md:hidden"
      aria-label="hamburger-navigation"
      on:click={hamburgerEventHandler}
    >
      <MoreVertical />
    </button>

    <!-- navigation menu -->
    <nav
      class="bg-white hidden md:block drop-shadow-md md:drop-shadow-none absolute md:relative right-2 top-full p-2 md:p-0"
      class:hidden={!isHamburgerClicked}
    >
      <div class="flex flex-col md:flex-row">
        <ul
          class="flex flex-col md:flex-row md:space-x-10 lg:space-x-20 text-sm"
        >
          {#each navigation as nav}
            <li class="group relative">
              <a
                class="no-underline inline-block p-2 hover:bg-gray-100"
                href={nav.link}
              >
                {nav.name}
              </a>
              {#if nav.children}
                <ul
                  class="z-10 bg-white text-left whitespace-nowrap md:hidden md:group-hover:block md:absolute md:drop-shadow-lg"
                >
                  {#each nav.children as child}
                    <li>
                      <a
                        class="no-underline inline-block py-2 px-4 w-full hover:bg-gray-100"
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
        <div class="flex justify-center items-center gap-2 ml-6">
          <a
            href="https://github.com/World4AI"
            target="_blank"
            rel="noreferrer"
          >
            <Github size={20} />
          </a>
          <a
            href="https://twitter.com/World4AI"
            target="_blank"
            rel="noreferrer"
          >
            <Twitter size={20} />
          </a>
        </div>
      </div>
    </nav>
  </div>
</header>
