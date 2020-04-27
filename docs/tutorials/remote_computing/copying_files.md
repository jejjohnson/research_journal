# Copying Files


## From Server to Desktop

### SCP

A lot of times you'll get coworkers who can't access or they don't use (or don't want to learn) how to use the server effectively. So they might ask you to help them transfer some files. One way to do it is to use the `scp` package. The command I use is below.

```bash
scp -r user@your.server.example.com:/path/to/foo /home/user/Desktop/
```

This works fairly well. And then I'll use [wetransfer](https://wetransfer.com/) to send and recieve the files. Done.