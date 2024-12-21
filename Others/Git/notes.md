# Git

The following notes come from the following resources:

- [Learn git branching](https://learngitbranching.js.org/)
- [Oh my git](https://ohmygit.org/)
- General internet searching of terms

## Branches

A branch is nothing more than a new pointer to a certain commit. Git branches are useful to create commits starting at a certain commit `C0`, so that any commit done after will have `C0` and its ancestors. Whatever branch you have "selected" when making a commit, is the branch that you will be updating its pointer for to that new commit. The selected branch has an asterisk \*

- To create a branch: 
	- `git branch <branch>`
- To start making commits into a branch: 
	- `git checkout <branch>`
- Or, even do both at the same time: 
	- `git checkout -b <branch>`

## Combining

We can **merge** branches with a "merge commit", which applies all changes from the source (feature branch) to the target in that commit. This commit has two parents corresponding to the last commit of both branches before the merge. So, the "merge commit" will have as ancestors all commits in both branches. The branch pointer `A` that advances to the "merge commit" is the one that was selected when doing the merge, because the `B` branch is being merged **INTO** `A` branch.

- To merge two branches, merging `B` into `A`, advancing the pointer in `A`: 
	- `git checkout A; git merge B;`

> In general, we don't want to advance `B`'s pointer into `A` after merging it into the destination branch (we don't want to fast-forward the source branch) because we can lose track of where our merge was located, lose information, etc. So we don't want to advance the `B` pointer as well when doing a merge.

The other way of combining work is using **rebase**. With it, we can **COPY** (not move) all of the commits exclusively done within `B` branch (after its base) onto a new base (commit from which we branched off from) which is going to be the pointer (last commit) of `A` branch. This will make all the commits seemingly linear, as a part of `A`. *i.e.: it's copy paste `B` into the end of `A`.*

- To rebase `B` into `A`, placing the pointer of `B` at the very end:
	- `git checkout B; git rebase A;`

> Contrary to merge, it is standard to advance the destination (main) branch `A` to the new location of the other branch `B`. This is done so that everyone on the production environment `A` can now update to changes introduced through `B`. To advance main we can simply just do `git checkout A; git merge B`. Since `A` is an ancestor of `B` now, we just fast-forward `A`.

But, which one to choose?

> *Merging is a safe option that preserves the entire history of your repository, while rebasing creates a linear history by moving your feature branch onto the tip of main .*

## Conflicts

When there is a conflict between two branches (a conflict between the state of a file AT the head pointers of two branches) in a merge or rebase operation, we will need to select the changes we want to apply. We can select between either or modify the file as convenient to resolve the conflict.

Conflict resolution changes / commits differ with the operation done:

- In a merge, the changes are applied directly in the new merge commit, and thus we leave all previous commits unchanged. This is why, just as with any other commit, we must resolve the files with changes, stage the files, and commit. The merge commit might as well be called a "resolution commit" in this case.
- In a rebase, the replaying ("copy pasting") process is done by going up through the other branch tip to its ancestors until it reaches its base, because we want the latest state of all files in the branch to encounter conflicts with. This is the order in which commits are replayed onto the main branch. When one conflict arises, the rebase process is paused until changes are introduced and staged. Then, when we `git rebase --continue`, the individual copy pasted commit will modify its original changes to match the resolution ones. When getting to the base, we are done with the combination of branches.

## Head

The HEAD is "hidden" inside the latest commit of our branches. The HEAD is the actual thing which acts as a pointer to the state that we are currently visualizing of the repo, so it is like the current commit. This current state of visualization is called the **working tree**, because it represents the state of the tracked files in a repo at a certain commit. 

By default, HEAD advances with new commits in a branch, maintaining its position at the branch's head. A head can also be `git checkout` into a specific commit instead of a branch:

- To detach HEAD into a commit `commit2`:
	- `git checkout commit2`
- To go back to main's head (last commit):
	- `git checkout main`

Now, instead of typing an entire commit as an absolute reference, we can use relative references:

- `HEAD^` to go up a single commit at a time. e.g.: `HEAD^^^`
- `HEAD~n` to go up `n` commits at a time. e.g.: `HEAD~3`

To move a branch `A` (its pointer) to the parent of another commit `c1` using relative references:

- `git checkout c1`
- `git branch -f A c1^`
- `git checkout A`
 
Combining relative references with multiple pointers (from branches and HEAD) is powerful.

## Reversing

- `git reset HEAD^` literally moves the current branch (and HEAD) pointers to the commit specified. Better for local / individual repos
- `git revert HEAD` creates a new commit which introduces the inverse changes to the changes in the `HEAD` commit. Thus, this new "reversing commit" can be pushed to a repo.
- `git checkout` moves the HEAD pointer to a branch head or previous commit, so the working directory is changed to the state of that past commit but doesn't actually revert any changes. It's like reverting but only the working directory to "check a commit out".

## Moving

To move commits around, like cherry picking commits from somewhere to another place, we can literally use `git cherry-pick <commit>`. This will introduce the changes of the commit into the current branch. By this is meant that the commits listed will be copy and pasted to the current branch's head. Like rebase but for specific commits (from any branch).

The other, more powerful, way of moving specific commits from a `B` branch until its base is by rebasing the branch into another one `A` (main) in interactive mode via `git rebase -i A`. This opens a vim GUI where we have multiple options to do, such as pick some commits, drop some, edit the commit message, and [more advanced stuff](https://git-scm.com/docs/git-rebase#_interactive_mode).

## Deleting

When we delete a branch we are just deleting a pointer. The commits done still remain intact as long as they aren't orphaned. Orphaned commits are commits that aren't reachable by any pointer (by any branch or by HEAD), which means that no branch's ancestors match the orphaned commits. Orphaned commits are recurrently deleted by git with its garbage collection process.

On the other hand, deleting a branch right after merging it with another will keep the commits on the history of the target branch, and thus the commits won't be orphaned.

## Index

![](assets/Pasted%20image%2020240413223827.png)

The index is like a preview of what is going to be committed. It also called the staging area. When you change a file in the working directory and don't update the index, then those changes aren't going to be reflected in case you commit your changes since we only keep track of changes directly added / updated in the index.

If we want to select specific changes within a file to add to the index, we can use `git add -p`, or git add patch. This way we can review each "patch" of changes (small groups of changes) one at a time and decide to add it or not to the index. The options to select are:

- `y` for yes add
- `n` for no add
- `s` to split the batch even more
- `e` to edit the patch
- `?` for help

