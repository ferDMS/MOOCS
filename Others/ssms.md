# SQL Server Management Studio (SSMS)

## Object Explorer

You can have multiple connections from different sources of SQL Server type services which can vary in functionalities and data stored. Each service has a different icon.

Inside the properties of the DB you can see the DB's server collation which describes the specific installation for SQL Server that is running the DB. `CI` stands for case insensitive. Properties also have information about the hardware running the DB.

The `Databases` directory is the one used by developers to see information about the DBs on the server, while other folders are mostly managed by DBAs (Database Administrators).

The object explorer has to be updated whenever a new table is created or something is edited that should show up but doesn't. It can be done with the blue arrow. The refreshing is only done on the context of the object explorer, meaning selected directory.

## Query Window

On the bottom is the connection status, server name, user and database context. This last one is the DB in which we will apply the queries. On an Azure DB because the DB is maintained independently and we aren't maintining an SQL Server instance, we can only see `master` and the current DB. That can be seen top left.

Execute query with f5. Error messages are found on the `Messages` tab, as well as succesful messages.

Highlight a statement to only execute that one statement.

To insert snippets you can use `Ctrl+K+Ctrl+X`. These snippets allow to create quick queries and other type of scripts like stored procedures that come in a template for quick editing.

Any window can be split via the slider to the top right.

## Obj. Explorer + Query W.

You can drag elements from the object explorer into the query window to append the specific name needed to call that specific object. For example, by pulling a table we would get the table's name. You can also drag the `Columns` directory to drag all columns of a table at once, in the form of comma separated column names.

You can create scripts for multiple tasks on the query window for any selected table on the object explorer. Right click, `Script Table as`, and select any kind of operation. This way we can recreate a script to `CREATE TABLE` or for `SELECT`, `UPDATE`, etc.

You can dock windows in multiple places whenever you need to be working with multiple files at a time.

## Editor Window

The editor window is like spawning an editable spreadsheet. It can be opened for any table through right click and then `Edit Top 200 Rows`. Inside of this interactive and dynamic window one can edit values. To change the view of the available-to-edit tuples,  select `SQL Pane`  or `Ctrl+3` , which opens a query window specific for the Results Pane (that should be executed with `Ctrl+R`).

## View Designer Window

By creating a new view or editing one you can access a designer window. In this window you can access the same tools mentioned above. `Ctrl+1` for the diagram panel, which is a ERM representation of any tables that you want to include. Inside of this panel you can add and remove tables and relationships between them (using connections with keys) as well as select output. All of the connections then generate a query on the SQL Panel to join the tables and select data.

## Stored Procedures and Functions

A function must return a value, must have at least one parameter, and is not allowed to change anything from the environment. A stored procedure can have 0 parameters, can have 0 return values, and can change database objects.

With a path of `DB/Programmability`, you can see the saved procedures and functions as well as create new ones and modify existent ones (like using an `ALTER` over the stored procedure).

Stored procedures can be executed using `Script Stored Procedure as` and selecting `Execute`. Using alter here would be the same as selecting modify.

## Security

The `dbo` user is the default local user for a DB in SQL Server. This is found inside the `Security` directory inside the database. Changes made in here apply only for the DB. Inside the `Security` directory for the Database Server we can find the users that can login to the server, which is where the Azure login user is found.

Being a `sysadmin` at server level allows you to enter all DBs inside the server.

## SQL Editor

The SQL Editor is the second row of tools found on the top. To the right of the execute button we can parse a query before sending it to check for correct syntax (`Ctrl+F5`).

Options here also include how to export the query result, which can be in text, grid (default), and .rpt file

## Templates

When using templates we can hit `Ctrl+Shift+M` to open a dynamic popup with the fields to fill in the template in an organized matter, including the type of data that has to be put in. This is really helpful as the templates are usually more verbose that needed. Careful that they can things that aren't wanted (like drop tables).

## Solutions and Proyects

A solution can contain multiple proyects inside of it. When using multiple script files we can create a proyect which can contain them all. Inside a proyect we can save connections and queries (which are .sql files) and other miscellaneous related files.
