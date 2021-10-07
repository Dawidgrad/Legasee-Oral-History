# Legasee Oral History
SLT Cohort 3 Team 2  Mini-Project

## How to use the repository & project management suggestions

Short, informal instruction on how we can use this repo along with some suggestions on how we can manage the project. Some of it will probably sound like common sense, nevertheless I wanted to put it in writing so we can refer back to it at any time. Feel free to suggest anything, maybe something I missed, or alternative way of going about it. Readme will probably be changed into something more sensible later. 

I thought that it would be potentially beneficial to follow a standard approach that is used in software development (well, at least stripped-down version of it). The branches are described below.

### Branches
 * main - Latest, stable version of the software that we know works,
 * develop - Branch that will be used to merge most recent changes into,
 * *{feature branch}* - e.g. legasee-1 ; These don't exist yet, just make one by branching off from develop when you are starting to work on some feature.
 
The **main** and **develop** branches are the most important in the repository. The **develop** branch is there to keep the main branch clean, which should contain nothing but release-ready code (well, we won't be making releases, but hopefully you get the idea). **Feature branches** on the other hand will allow everyone to work on multiple features separately, at the same time. Commiting straight to **develop** will in most cases throw others' code off base (code on the develop branch locally not matching the online version) and might result in merge conflicts. They will also allow us to review code and potentially revert any changes easier.

### Kanban board

Ideally, we will break down bigger tasks into smaller ones and put those on the Kanban board (like the one on Trello). I have a suggestion as to how one might look like. We could utilise 4 columns:
 * Backlog - The tasks that we will work on in the future,
 * To do - The tasks that we are going to work on this week / these 2 weeks,
 * In progress,
 * Done.
 
Each of the tasks that we have in the **To do** column should ideally have a clear description of what needs to be done. The tasks in the **Done** column should contain all of the information on what has been done etc. just so the others can check and see what has been changed / found. 

The example that I just created to visualise it:

![Kanban board](https://i.imgur.com/iiNaW9e.png)

Tom: The "Board" view in ClickUp behaves like this - I believe it's designed to replicate exactly this kind of Trello board, will hopefully do the trick.

### Task lifecycle 

I can describe the process that I imagined we could potentially follow (at least for those tasks that require implementation).

*Optional: We could decide on the rough scope of the tasks in backlog during our meetings and move it into To do column.*
1. Move the task from To do, into In Progress on the Kanban board and assign it to yourself.
2. Create a branch with the name of the task (having some identifiers would be helpful with tracking the changes, e.g. legasee-1, legasee-2).
3. Work on the feature on that branch.
4. Create a pull request (e.g. legasee-1 -> develop) once you have completed the feature and ensured that there are no errors.
5. Merge into develop.
6. Move task into done column on Kanban board and make sure that it's described.
7. Once every week or two we could merge develop into the main branch, once we know that develop works fine. (Automated tests?).

We could potentially require an approval of the pull request by other person (between step 4 and 5), just so there is another set of eyes looking for any issues before merging into develop. This will probably be an overkill though, since it's not a software engineering project as Stu has mentioned.

Any questions or suggestions, just let me know! :) Feel free to edit this for clarity too.
