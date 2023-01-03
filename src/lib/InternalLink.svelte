<script>
  export let type;
  export let id;
</script>

<a
  id="{type}-origin-{id}"
  class="text-black text-xs align-top font-bold scroll-mt-24 no-underline"
  href="#{type}-{id}"
>
  [{id}]
</a>
