# pull info daily
- have this run on start up.
> or really however you want to automate running this
> I commented out a 24 hour sleep if we want to leave this running on a server somewhere

- it will check the database to see if it ran today
- assuming it has not it will then hit the plolinex api
- add the response to a database

> this will crash if there is no database in the folder
> when it checks to see if the script ran today if there is no database to check it will error out
> I could fix this but didnt because lazy
