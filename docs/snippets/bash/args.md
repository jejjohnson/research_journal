# Arguments in Scripts


```bash
#!/bin/bash
     
### Print total arguments and their values
    
echo "Total Arguments:" $#
echo "All Arguments values:" $@
    
### Command arguments can be accessed as
    
echo "First->"  $1
echo "Second->" $2
    
# You can also access all arguments in an array and use them in a script.
    
args=("$@")
echo "First->"  ${args[0]} 
echo "Second->" ${args[1]}
```