import re
import subprocess
import datetime
from peewee import SqliteDatabase
from peewee import Model

# from peewee import * # maybe change later...
# SQLite database using WAL journal mode and 64MB cache.
from peewee import (
    IntegerField,
    BigIntegerField,
    SmallIntegerField,
    CharField,
    TimestampField,
    TextField,
    BooleanField,
    DateTimeField,
    ForeignKeyField,
    FloatField,
)


def get_database_conn(path=None, conn=None):
    # TODO allow config
    database = SqliteDatabase(":memory:", pragmas={"foreign_keys": 1})
    return database


database = get_database_conn()


class Base(Model):
    class Meta:
        database = database


class Instrument(Base):

    name = CharField(unique=True)
    qc_recno = BigIntegerField(null=True)
    last_runno = IntegerField(null=True)
    last_timestamp = TimestampField(resolution=1)
    qc_dir = TextField(null=True)


class Label(Base):
    name = CharField()


class Experiment(Base):

    recno = BigIntegerField(unique=True)
    projectno = BigIntegerField(null=True)
    label = ForeignKeyField(Label, default="experiments", null=True)

    def __repr__(self):
        return f"Experiment {self.recno}"


class ExpRun(Base):

    runno = SmallIntegerField()
    recno = ForeignKeyField(Experiment, backref="runs")
    timestamp = DateTimeField(default=datetime.datetime.now)


class ExpSearch(Base):
    searchno = SmallIntegerField()
    recno = ForeignKeyField(Experiment, backref="searches")
    masicflag = BooleanField(default=0, null=True)
    searchflag = BooleanField(default=0, null=True)
    grouperflag = BooleanField(default=0, null=True)


class RawFile(Base):
    # fullpath = CharField(max_length=255, unique=True)
    filename = CharField(max_length=255)
    exprec = ForeignKeyField(Experiment, backref="rawfiles")
    exprun = ForeignKeyField(ExpRun, backref="rawfiles")
    birth = DateTimeField(null=True)
    size = FloatField(null=True)
    instrument = CharField(null=True)


class PSM(Base):

    # scan = BigIntegerField()
    charge = SmallIntegerField()
    native_id = CharField(max_length=100, default="none")
    precursor_neutral_mass = FloatField()
    rt_sec = FloatField()
    peptide = CharField(max_length=100)
    modified_peptide = CharField(max_length=100)
    num_missed_cleavages = SmallIntegerField()
    hit_rank = SmallIntegerField()
    tot_num_ions = SmallIntegerField()
    num_matched_ions = BigIntegerField()
    mass_diff = FloatField()
    is_rejected = BooleanField()
    search_score = FloatField()
    probability = FloatField()
    calc_neutral_pep_mass = FloatField()

    recno = ForeignKeyField(Experiment, backref="psms")
    runno = ForeignKeyField(Experiment, backref="psms")


def create_tables(database=database):
    # with database:
    database.create_tables(
        [Experiment, ExpRun, ExpSearch, Label, PSM, Instrument, RawFile]
    )


create_tables()


def populate_instruments():

    """
    populate Instrument database. These are the field:
    name : name of instrument
    qc_recno : recno of instrument qc
    last_runno : last runno that QC was performed
    last_timestamp = last timestamp at which QC was checked (used to find new raw files)
    """

    instruments = [
        {
            "name": "Orbitrap Fusion",
            "qc_recno": 99999,
            "qc_dir": "/mnt/win_share/amms00/d/AMMS00_Data/",
        },
        {
            "name": "Orbitrap Exploris Slot #0091",
            "qc_recno": 99990,
            "qc_dir": "/mnt/win_share/FPKNNR2/data/QC/",
        },
        {"name": "Q Exactive Plus Orbitrap", "qc_recno": 99994, "qc_dir": ""},
        {
            "name": "Ellis Orbitrap Fusion Lumos",
            "qc_recno": 99998,
            "qc_dir": "/mnt/win_share/Ellis_Lumos/d/QC/",
        },
        {
            "name": "Lumos_ETD",
            "qc_recno": 99995,
            "qc_dir": "/mnt/win_share/Lumos_ETD/d/QC/",
        },
    ]

    with database.atomic():
        Instrument.insert_many(instruments).execute()
