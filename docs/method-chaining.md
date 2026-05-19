# Method chaining

Good internal DSLs in Python enable method chaining:

```python
pd.DataFrame(data).T.describe().to_html()
np.array(data).square().mean().sqrt() * .3
str(txt).lower().replace(',', ' , ').split()
```

We could even inherit some DSL container base class that already does chaining, such as a Pandas `DataFrame`.

Our base class might be called `ThoughtChain` or `Reasoning`.
And the 2nd level class that inherits Reasoning  directly might be Turns or Conversation and could also inherit from the Conversation class defined in the OpenAI API specification.
So your first reasoning processing step would be to break down the conversation into smaller logical natural language phrases and append this as a new attribute to the object, a list of unordered tasks or a TaskList object. 
And the next step would be to create an OrderedTaskList or TaskPlan object, etc.

The method chaining might look like:  

```python
messages = [{
   prompt="find top 3 wikipedia mentions of Paracetemol"}]

Conversation(messages) (
    .analyze()  # SLM split NL task phrases
    .plan()  # reorder task (dependencies)
    .select_tools()  # NL => fun(kwargs)
    .optimize()  # reorder, combine
    .check_maliciousness() # privacy/security
    .check_safety() # prosocialness, ethics
    .check_permissions() # security rules
    .check_whitelist() # allowed tools+opts
    .check_blacklist() # disallowed tools
    .run()  # use tools in order planned
    .evaluate()  # outcome quality score
    .explain()  # describe outcome NL msg
```
