# ------------------------------------------------------------------------
from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import numpy as np
import re
import string
import spacy

import matplotlib.pyplot as plt
nlp = spacy.load("en_core_web_sm")
#from collections import Counter
stopwords = '''x
y
your
yours
yourself
yourselves
you
yond
yonder
yon
ye
yet
z
zillion
j
u
umpteen
usually
us
username
uponed
upons
uponing
upon
ups
upping
upped
up
unto
until
unless
unlike
unliker
unlikest
under
underneath
use
used
usedest
r
rath
rather
rathest
rathe
re
relate
related
relatively
regarding
really
res
respecting
respectively
q
quite
que
qua
n
neither
neaths
neath
nethe
nethermost
necessary
necessariest
necessarier
never
nevertheless
nigh
nighest
nigher
nine
noone
nobody
nobodies
nowhere
nowheres
no
noes
nor
nos
no-one
none
not
notwithstanding
nothings
nothing
nathless
natheless
t
ten
tills
till
tilled
tilling
to
towards
toward
towardest
towarder
together
too
thy
thyself
thus
than
that
those
thou
though
thous
thouses
thoroughest
thorougher
thorough
thoroughly
thru
thruer
thruest
thro
through
throughout
throughest
througher
thine
this
thises
they
thee
the
then
thence
thenest
thener
them
themselves
these
therer
there
thereby
therest
thereafter
therein
thereupon
therefore
their
theirs
thing
things
three
two
o
oh
owt
owning
owned
own
owns
others
other
otherwise
otherwisest
otherwiser
of
often
oftener
oftenest
off
offs
offest
one
ought
oughts
our
ours
ourselves
ourself
out
outest
outed
outwith
outs
outside
over
overallest
overaller
overalls
overall
overs
or
orer
orest
on
oneself
onest
ons
onto
a
atween
at
athwart
atop
afore
afterward
afterwards
after
afterest
afterer
ain
an
any
anything
anybody
anyone
anyhow
anywhere
anent
anear
and
andor
another
around
ares
are
aest
aer
against
again
accordingly
abaft
abafter
abaftest
abovest
above
abover
abouter
aboutest
about
aid
amidst
amid
among
amongst
apartest
aparter
apart
appeared
appears
appear
appearing
appropriating
appropriate
appropriatest
appropriates
appropriater
appropriated
already
always
also
along
alongside
although
almost
all
allest
aller
allyou
alls
albeit
awfully
as
aside
asides
aslant
ases
astrider
astride
astridest
astraddlest
astraddler
astraddle
availablest
availabler
available
aughts
aught
vs
v
variousest
variouser
various
via
vis-a-vis
vis-a-viser
vis-a-visest
viz
very
veriest
verier
versus
k
g
go
gone
good
got
gotta
gotten
get
gets
getting
b
by
byandby
by-and-by
bist
both
but
buts
be
beyond
because
became
becomes
become
becoming
becomings
becominger
becomingest
behind
behinds
before
beforehand
beforehandest
beforehander
bettered
betters
better
bettering
betwixt
between
beneath
been
below
besides
beside
m
my
myself
mucher
muchest
much
must
musts
musths
musth
main
make
mayest
many
mauger
maugre
me
meanwhiles
meanwhile
mostly
most
moreover
more
might
mights
midst
midsts
h
huh
humph
he
hers
herself
her
hereby
herein
hereafters
hereafter
hereupon
hence
hadst
had
having
haves
have
has
hast
hardly
hae
hath
him
himself
hither
hitherest
hitherer
his
how-do-you-do
however
how
howbeit
howdoyoudo
hoos
hoo
w
woulded
woulding
would
woulds
was
wast
we
wert
were
with
withal
without
within
why
what
whatever
whateverer
whateverest
whatsoeverer
whatsoeverest
whatsoever
whence
whencesoever
whenever
whensoever
when
whenas
whether
wheen
whereto
whereupon
wherever
whereon
whereof
where
whereby
wherewithal
wherewith
whereinto
wherein
whereafter
whereas
wheresoever
wherefrom
which
whichever
whichsoever
whilst
while
whiles
whithersoever
whither
whoever
whosoever
whoso
whose
whomever
s
syne
syn
shalling
shall
shalled
shalls
shoulding
should
shoulded
shoulds
she
sayyid
sayid
said
saider
saidest
same
samest
sames
samer
saved
sans
sanses
sanserifs
sanserif
so
soer
soest
sobeit
someone
somebody
somehow
some
somewhere
somewhat
something
sometimest
sometimes
sometimer
sometime
several
severaler
severalest
serious
seriousest
seriouser
senza
send
sent
seem
seems
seemed
seemingest
seeminger
seemings
seven
summat
sups
sup
supping
supped
such
since
sine
sines
sith
six
stop
stopped
p
plaintiff
plenty
plenties
please
pleased
pleases
per
perhaps
particulars
particularly
particular
particularest
particularer
pro
providing
provides
provided
provide
probably
l
layabout
layabouts
latter
latterest
latterer
latterly
latters
lots
lotting
lotted
lot
lest
less
ie
ifs
if
i
info
information
itself
its
it
is
idem
idemer
idemest
immediate
immediately
immediatest
immediater
in
inwards
inwardest
inwarder
inward
inasmuch
into
instead
insofar
indicates
indicated
indicate
indicating
indeed
inc
f
fact
facts
fs
figupon
figupons
figuponing
figuponed
few
fewer
fewest
frae
from
failing
failings
five
furthers
furtherer
furthered
furtherest
further
furthering
furthermore
fourscore
followthrough
for
forwhy
fornenst
formerly
former
formerer
formerest
formers
forbye
forby
fore
forever
forer
fores
four
d
ddays
dday
do
doing
doings
doe
does
doth
downwarder
downwardest
downward
downwards
downs
done
doner
dones
donest
dos
dost
did
differentest
differenter
different
describing
describe
describes
described
despiting
despites
despited
despite
during
c
cum
circa
chez
cer
certain
certainest
certainer
cest
canst
cannot
cant
cants
canting
cantest
canted
co
could
couldst
comeon
comeons
come-ons
come-on
concerning
concerninger
concerningest
consequently
considering
e
eg
eight
either
even
evens
evenser
evensest
evened
evenest
ever
everyone
everything
everybody
everywhere
every
ere
each
et
etc
elsewhere
else
ex
excepted
excepts
except
excepting
exes
enough'''
stopwords = stopwords.split("\n")
#from jinja2 import escape

# --------------------------------------------------------------------------

app = Flask(__name__)


@app.route('/')
def index_page():
    return render_template("index.html")


@app.route('/info', methods=["POST"])
def info():
    name_ = request.form.get("uname")

    tf = pickle.load(open("tfidf.pkl", "rb"))
    model = pickle.load(open("mn.pkl", "rb"))

# ------------------------FUNCTION---------------------------------------
    def word_count(x):
        return len(x.split(" "))

    def punc_remove(x):
        for i in x:
            if i in string.punctuation:
                x = x.replace(i, "")
        return x


    def Number(x):
        return str(re.sub("[0-9]+", "", x))


    def make_lemma(x):
        sent = []
        docs = nlp(x)
        for i in docs:
            sent.append(i.lemma_)
        return " ".join(sent)


    def RemoveStopWords(x):
        my_sent = []
        for i in x.split(" "):
            if i not in stopwords:
                my_sent.append(i)
        return " ".join(my_sent)


    def RemovesingleChar(x):

        my_sent = []
        for i in x.split(" "):
            if len(i) > 1:
                my_sent.append(i)
        return " ".join(my_sent)


# def SelectMeaningfullWords(x):
#     sentences = []
#     docs = list(nlp.pipe(x))
#     for i in docs:
#         sent = []
#         for j in i:
#             if j.has_vector == True:
#                 sent.append(j.lemma_)
#         sentences.append(" ".join(sent))
#     return sentences


    def alpha(x):
        match = re.sub('[@_!#$%^&*()<>?/\|}{~:]', "", x)

        return str(match)


    def preprocessing(x):
        x = x.lower()
        x = punc_remove(x)
        x = alpha(x)
        x = Number(x)
        x = RemoveStopWords(x)
        x = make_lemma(x)
        x = RemovesingleChar(x)
        return x


    # ---------------------------------------------------------
    text = preprocessing(str(name_))
    
    #tran = tf.transform([text])
    #result_ = list(model.predict(transform).toarray())
    result_= model.predict(tf.transform([text]).toarray())
   
    if result_[0]==0:
        return render_template("result.html",name="Not Spam")
    else:
        return render_template("result.html",name="Spam")
        


if __name__ == "__main__":
    app.run(debug=True)
