ECHO off
powershell -Command "Get-Service -Name ssh-agent | Set-Service -StartupType Manual"

SET file_key_name=id_rsa

(
echo -----BEGIN OPENSSH PRIVATE KEY-----
echo b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABlwAAAAdzc2gtcn
echo NhAAAAAwEAAQAAAYEAwwwjmstsOA5fvRpKpQ9ZyRCMBhotLOi05b0/NUSO3/bp7YQHBd4B
echo hY7xvYRWKWyC5xG527m+/vnlpNNwDGAanGi9iNq/2+Mbz6JzDwD3jnszsgbCgpi6ixUMQy
echo xe4wMYzoQjzbA2M6bKK8aeAWCcsRnM7kkA4QODNq0XIuLU9uD/NeZv8zLkdBXpqWrscOiT
echo 22GluIxi3s8acU418KJ266v6ag1LKi2jdzcU6rIE6mLrdKX0kPD5bE4TKaE7ytvRxJTj9M
echo n3eeOKYbVFRQXL5YPkryP3WHuVDtLSrB0N38BKrQ6ZOv67E69ZUjqrAoB6Ik2PidGNunra
echo y9i4nHxx/SzUxccjBp74RrezfkKbZio7ljjLlFVqcZiNf5t13jJuSobVFEiOFNjXmFCP3B
echo UjiQGDgQ5rMotYU5AMQkx689JnVZC9VtoxqsLqil7oIq+nVHn3jdH3AHXX5IVpgNDs7KNa
echo /sOn2aKGzW23r3MeZWKS9N8EV8DBrILlBFpePpabAAAFkFC4sX5QuLF+AAAAB3NzaC1yc2
echo EAAAGBAMMMI5rLbDgOX70aSqUPWckQjAYaLSzotOW9PzVEjt/26e2EBwXeAYWO8b2EVils
echo gucRudu5vv755aTTcAxgGpxovYjav9vjG8+icw8A9457M7IGwoKYuosVDEMsXuMDGM6EI8
echo 2wNjOmyivGngFgnLEZzO5JAOEDgzatFyLi1Pbg/zXmb/My5HQV6alq7HDok9thpbiMYt7P
echo GnFONfCiduur+moNSyoto3c3FOqyBOpi63Sl9JDw+WxOEymhO8rb0cSU4/TJ93njimG1RU
echo UFy+WD5K8j91h7lQ7S0qwdDd/ASq0OmTr+uxOvWVI6qwKAeiJNj4nRjbp62svYuJx8cf0s
echo 1MXHIwae+Ea3s35Cm2YqO5Y4y5RVanGYjX+bdd4ybkqG1RRIjhTY15hQj9wVI4kBg4EOaz
echo KLWFOQDEJMevPSZ1WQvVbaMarC6ope6CKvp1R5943R9wB11+SFaYDQ7OyjWv7Dp9mihs1t
echo t69zHmVikvTfBFfAwayC5QRaXj6WmwAAAAMBAAEAAAGBAMAeb9+kKXdZqTHR+N52rWCgHN
echo xR4leO68gzTVRBsF2ojyi12FkOIP+WGkUrWdc5nALQcfqdDiWGro1Y+tAlxXB0tuRbW4nS
echo PO2bPKv1ruI4NmZPxD3xBCXE2Kw9w0TmIwQgIkgTHoBn9FGENNR0fkLvf+ziGayJ0jAD9H
echo sXZN+8JYXI8lJhuacigmKvADAC1sjLePm7xhNjec9LU1QwIuTmJCVb7MKh65acucynBrJ8
echo NyJR6QDvay8kBDoIJ6YcN9wiAXdFWbXyjszpmtHSTWUuaW0/ZXhlDxpYbHoY8orU/Tv2Ry
echo CNsizL10SljAkcArFtfLGv1sbv4rq953yNJIFdgav1Z17pcHRpCC1e4cWLwtM8kkMm27fN
echo cJaAL+phPbjhftv+UPKqQAIQpPFHF+CsbfHiy7ep+muaMAKj+BTR15N7asTB7M/BOaEhtx
echo ATJ0DgmtbbFwr6vDa4DoQjylAIaIHVV2Bxn67JaPg88cWml4Pgsq/hS9VWlwkr7WiriQAA
echo AMEAxl5LG122QcTU3KQAFrKsmVfbo9uXb3UMI5RpEqwnjDCJXF3AeZwOvCU87YLRki+l4b
echo x0OIR/wIG9qZqQk9+qYii9HmCUZ5vu/5Z1HVEYha34jLzRdJ8kbZxmQ7nyncknAFpr1QpP
echo NhzaB5i6jzBA7YCb/WBThBnzwdRFtscXmdbOi/dBD+8Hx95yEZOXgB0yj5JoVVXhCsb/bb
echo +WfPn/X7BaYhjmb1Zqx+NuQNZ5ahd0jBqj12iRhpBT08aaoEy1AAAAwQDy/zGmAy+JXUPi
echo 71SSr2Gga3J8HU+mOIaXZkvRQkq6Px2bCB+tbmbBzm6yJpt9yBLV5LysygvgculsPqSjVj
echo 1uTCl8c7BuKULE4OmoFkD3JhxA0yJrei+fmbYKn+tKkhnzoxFZAv7fLfjJTZV4Xv840WMv
echo MjNG1tr+Fv8Z47pWdJlQhjqzCt4ck/EI5kc56IF9KyFvuCC85hoBKOPc44BSe4IPwBX4wU
echo yiG/ub6/5cu25phlXS+HCoE+7POF8r3I0AAADBAM18Fl+9GMWcOQWRHyZIgaVfU8N/lVLs
echo cWLbBU/oz3L3FL2m+8rUw13JoSRtcg9zDjXjKYCIzuILC57B89tfPH3VNT2VSA7OVG47Ah
echo O5pNM9bwdq1YZ9pp9G7uMCP+tjKVy+BQ6fcGRZ7ldWzUB3xq7lfBbtLnPjuJyCNbZNn7lI
echo Eitm0NPdJxO2rC4yhoeyJuRP6YgwRH7DSZ93jC3WrMvT2+Qg7GGciUeujLcE/lasXW5P+4
echo z+3sgnEcc9DmX5xwAAABRKb3NoQERFU0tUT1AtVjlJNjY4MgECAwQF
echo -----END OPENSSH PRIVATE KEY-----
) > %homedrive%%homepath%\.ssh\%file_key_name%

(
echo.
echo Host github.com
echo HoseName github.com
echo User git
echo IdentityFile ~/.ssh/%file_key_name%
) > %homedrive%%homepath%\.ssh\config

CALL "ssh-add" "%homedrive%%homepath%\.ssh\%file_key_name%"
CALL "ssh-keyscan" "github.com" >> "%homedrive%%homepath%\.ssh\known_hosts"

setlocal
set "psCommand="(new-object -COM 'Shell.Application')^
.BrowseForFolder(0,'Choose folder to clone Full_SMS into. A new folder called Full_SMS will be created.',0,0).self.path""
for /f "usebackq delims=" %%I in (`powershell %psCommand%`) do set "folder=%%I"
setlocal enabledelayedexpansion

CALL "git" "clone" "ssh://github.com/BioPhysicsUP/Full_SMS.git" "!folder!\Full_SMS"

pause