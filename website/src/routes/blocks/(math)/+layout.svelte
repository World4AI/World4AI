<script>
  import Sidebar from "$lib/Sidebar.svelte";
  const math = [
      {
        name: 'Introduction',
        link: '/blocks/mathematics/introduction', 
        links: [
        ]
      },
      {
         name: 'Calculus',
         link: '/blocks/mathematics/calculus',
         links: [
         ]
      },
      {
         name: 'Linear Algebra',
         link: '/blocks/mathematics/linear_algebra',
         links: [
         ]
      },
      {
         name: 'Probability and Statistics',
         link: '/blocks/mathematics/probability_statistics',
         links: [
         ]
      },
  ];
</script>

<Sidebar root={math} />
<main>
  <article>
    <slot />
  </article>
</main>
