# Good Code

- [Logging](#logging)
- [Testing](#testing)
- [Repository Organization](#repository-organization)
- [Packaging](#packaging)
- [Type Checking](#type-checking)
- [Continuous Integration](#continuous-integration)
- [WorkFlow](#workflow)


## Logging

This is something I often use whenever I am in the process of building software and I think there are some key things that need to be documented. It is often much better than print statements. I often do this when I'm not really sure if what I did is correct in the process. It's really important to log. Especially when you're doing server computing and you need a history of what was going on.


<details>
<summary><b>My Style</b></summary>

* INFO - General stuff of where I am at in the program so I can follow the control of flow
* DEBUG - Typically more about sizes/shapes of my matrices or possibly in the checks
* WARNING - Where things could go wrong but I have ignored this part due to some reason.

</details>

<details>
<summary><b>Tutorials</b></summary>

* Python Logging: A Stroll Through the Source Code - [RealPython](https://realpython.com/python-logging-source-code/)
* Python Logging Cheat Sheet - [gist](https://gist.github.com/jonepl/dd5dc90a5bc1b86b2fc2b3a244af7fc6)
* The Hitchhikers Guide to Python: Logging - [blog](https://docs.python-guide.org/writing/logging/)
* Good Logging Practice in Python - [blog](https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/)
* Logging CookBook - [Python Docs](https://docs.python.org/3/howto/logging-cookbook.html)
* Corey Schafer    
  * Logging Basics - Logging to Files, Setting Levels, and Formating - [Youtube](https://www.youtube.com/watch?v=-ARI4Cz-awo&list=PLMdgUBu5wWKxObYWmWbwxDhlBXqUObLNY&index=2&t=0s)
  * Logging Advanced: Loggers, Handlers, and Formatters - [Youtube](https://www.youtube.com/watch?v=jxmzY9soFXg&list=PLMdgUBu5wWKxObYWmWbwxDhlBXqUObLNY&index=4&t=0s)

</details>

## Testing

Something that we all should do but don't always do. It's important for the long run but it seems annoying for the short game. But overall, you cannot go wrong with tests; you just can't.


<details>
<summary><b>My Style</b></summary>

* Package - PyTest
* IDE - VSCode

</details>


<details>
<summary><b>Tutorials</b></summary>

* Testing Python Applications with PyTest - [Blog](https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest)
* Getting Started with Testing in Python - [RealPython](https://realpython.com/python-testing/)
* Testing Your Python Code with PyTest - [SciPy 2019](https://www.youtube.com/watch?v=LX2ksGYXJ80)
* Learn PyTest in 60 Minutes: Python Unit Testing Framework - [Youtube](https://www.youtube.com/watch?v=bbp_849-RZ4)
* Python Testing in VSCode - [VSCode Docs](https://code.visualstudio.com/docs/python/testing)
* Eric Ma 
  * Testing Data Science Code - [YouTube](https://www.youtube.com/watch?v=fmVbtHMHEZc)
  * Best Testing Practices - [PyCon 2017](https://www.youtube.com/watch?v=yACtdj1_IxE)

</details>


## Repository Organization

## Packaging

**Source**:
* [PyPA](https://packaging.python.org/)
* [How to Package Your Python Code](https://python-packaging.readthedocs.io/en/latest/)


## Type Checking


## Continuous Integration


## WorkFlow

1. JupyterLab - Prototyping, Remote Computing
2. VSCode - Package Management, Remote Computing
3. Remote Computing - SSH (JLab, VSCode)