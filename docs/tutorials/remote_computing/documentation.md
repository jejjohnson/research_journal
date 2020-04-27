# Documentation

> **Having** good documentation is amazing but **doing** documentation sucks... - Me 

It's something we **should** do but we really don't do. I actually do write a lot of comments in my code (*sometimes*) but I never really present it anywhere for people to see. It's mainly for me so that I can remember what in the world I was doing. I do type-hints and I also use the `numpy`-style documentation so I'm forced to be a bit verbose in my explanations. But again, actually porting the code to documentation is the problem. In addition, when I'm coding stuff, there is often a lot of theory and mathematics that go behind the scenes. So I need somewhere to put it where it looks nice. Below, I outlined 3 requirements for what I wanted.

**Markdown**: I know it, I love it, and I don't want to use Restructured Text. I find the syntax not very pleasant.

**Easy** - I don't want to work to hard on this as I have other things I want to work on.

**Nice Format** - I like things to look nice. So anything where the outcome is quite attractive is nice for me.

Given these requirements, I have outlined below my process for doing documentation.

---

## Python Documentation - **pdoc**

So this is the package I use to actually port my comments to actual markdown files. I like the package [pdoc](https://pdoc3.github.io/pdoc/). So far, I have found it to be the **easiest** way to generate documentation from your python files to markdown files. It just works. It's also easy to use over an ssh connection so really removes some of the computational burden I have with my laptop. I have a few useful commands below and near the bottom of the document, I talk about installation.

---

### Commands

**Development**

For development, use this command. The `--http` command is so that I can edit my files remotely.

```bash
pdoc PKG --output-dir DIR --http HOST:PORT
```

**Deploying**

For deployment, I do the exact same thing except I don't actually use the `--http` argument because there is no need to view it.

```bash
pdoc PKG --output-dir DIR
```

---

## Webpages - **MkDocs**

This is what I use for generating the documentation. I find it has the most features for the least amount of work. In particular I use the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) version. Almost everything you need is in the configuration file and the rest is just the extensions. The mathjax took a bit to get used to but other than that, everything works fine.

### Commands

**Development**

```bash
mkdocs serve --dirtyreload -a localhost:3001
```

**Deploying**

```bash
mkdocs gh-deploy
```

??? tip "My .yml Config File"

    ```bash
    # Project information
    site_name: Research Journal
    site_description: My Personal Research Journal
    site_author: J. Emmanuel Johnson
    site_url: https://jejjohnson.github.io/research_journal

    # Repository
    repo_name: jejjohnson/research_journal
    repo_url: https://github.com/jejjohnson/research_journal

    # Configuration
    theme:
    name: material
    language: en
    palette:
        primary: black
        accent: gray
    font:
        text: source code pro
        code: source code pro

    plugins:
    - search
    - mknotebooks:
        execute: false
        write_markdown: true
        timeout: 600

    # Copyright
    copyright: Copyright &copy; 2020 J. Emmanuel Johnson

    markdown_extensions:
    - markdown.extensions.admonition
    - markdown.extensions.attr_list
    - markdown.extensions.codehilite:
        guess_lang: false
    - markdown.extensions.def_list
    - markdown.extensions.footnotes
    - markdown.extensions.meta
    - markdown.extensions.toc:
        permalink: true
    - pymdownx.arithmatex
    - pymdownx.betterem:
        smart_enable: all
    - pymdownx.caret
    - pymdownx.critic
    - pymdownx.details
    - pymdownx.emoji:
        emoji_index: !!python/name:pymdownx.emoji.twemoji
        emoji_generator: !!python/name:pymdownx.emoji.to_svg
    - pymdownx.highlight:
        linenums_style: pymdownx-inline
    - pymdownx.inlinehilite
    - pymdownx.keys
    # - pymdownx.magiclink:
    #     repo_url_shorthand: true
    #     user: squidfunk
    #     repo: mkdocs-material
    - pymdownx.mark
    - pymdownx.smartsymbols
    - pymdownx.snippets:
        check_paths: true
    - pymdownx.superfences
    - pymdownx.tabbed
    - pymdownx.tasklist:
        custom_checkbox: true
    - pymdownx.tilde

    extra_javascript:
        - javascripts/extra.js
        - https://polyfill.io/v3/polyfill.min.js?features=es6
        - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js
        # - https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML

    extra:
    # disqus: XHR39t5kZv
    social:
        # - type: 'envelope'
        #   link: 'http://www.shortwhale.com/ericmjl'
        - icon: fontawesome/brands/github
        link: 'https://github.com/jejjohnson'
        - icon: fontawesome/brands/twitter
        link: 'https://twitter.com/jejjohnson'
        - icon: fontawesome/brands/linkedin
        link: 'https://linkedin.com/in/jejjohnson'
        - icon: fontawesome/solid/globe
        link: 'https://jejjohnson.netlify.com'
    
    ```

    

---

## My Setup

Below are my specifics on how I setup the documentation on my system.

### Installation

I have everything installed under one package. I've never had any dependency issues so I never saw any reason to have separate environments.

```bash
conda create --name my_docs python=3.8
conda activate my_docs
pip install mkdocs-material mknotebooks pymdown-extensions pdoc3
```

---

### Folder Structure

Typically I like to have my documentation segmented into 2 main parts: my python package documentation and my specific notes. You'll notice in the tree structure below the **PKG** as a subdirectory of the **docs**. The files generated within there are from the `pdoc` command. The other subdirectories within the **docs** folder are things that I control myself using the `mkdocs`. That's how I structure it. I need something automated for the python documentation as well as the 


```bash
├── docs
│   ├── PKG
│   ├── Examples
│   ├── Theory
│   ├── Walk-Throughs
...
```

---

### Preferred Style

I personally like the `numpy` documentation style. It takes up quite a lot of vertical space so it makes your code documents a lot longer than they need to be. But I like it because it forces me to really try to explain what in the world is going on. Documentation is good but a lot of times I go digging into the code itself to understand what's going on. And personally I absolutely hate clutter documentation (e.g. when you have 10 arguments and they're all on top of each other with no space in-between the lines, so basically the `ReStructuredText` style). The `Google-style` is a bit more spacious but still, the arguments are often on top of each other which is still a bit harder to read for me. So, given that I've explained my preferences, I use the numpy style.

??? details "Example: Numpy-Style Documentation"
    Here is an example:

    ```python
    def function(arg1: int, arg2: float) -> None:
        """Summary Line

        Extended description of the function.

        Parameters
        ----------
        arg1 : int
            Description of argument 1
        
        arg2 : int
            Description of argument 2

        Returns
        -------
        score : float
            Description of return value

        See Also
        --------
        otherFunc: a related function

        Examples
        --------
        A good (or quick) illustration of how to use the function

        >>> arg1 = 2
        >>> arg2 = 10.
        >>> function(arg1, arg2)
        output

        """
    ```

You can find a [full example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy) from the Sphinx webpage. More examples with the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) package for Sphinx.

!!! tip 
    Because it's a pain to do documentation, I usually do it **first** before I actually code the function. Obviously I will have to change it later. But it actually forces to me think about what I want the function to do before I start coding. It's a sort of...forces me to think about the preliminary requirements or scope of my function beforehand. If I find that I'm writing too many things, then maybe I should shorten the function to do something smaller and more manageable.

---