from sqlalchemy import *
from migrate import *


from migrate.changeset import schema
pre_meta = MetaData()
post_meta = MetaData()
user = Table('user', post_meta,
    Column('id', Integer, primary_key=True, nullable=False),
    Column('name', String(length=64)),
    Column('email', String(length=120)),
)

reading = Table('reading', pre_meta,
    Column('id', INTEGER, primary_key=True, nullable=False),
    Column('user', VARCHAR(length=64)),
    Column('email', VARCHAR(length=120)),
    Column('device', VARCHAR(length=120)),
    Column('timestamp', VARCHAR(length=120)),
    Column('acc_x', INTEGER),
    Column('acc_y', INTEGER),
    Column('acc_z', INTEGER),
)

reading = Table('reading', post_meta,
    Column('id', Integer, primary_key=True, nullable=False),
    Column('device', String(length=120)),
    Column('timestamp', String(length=120)),
    Column('alt_timestamp', DateTime),
    Column('acc_x', Integer),
    Column('acc_y', Integer),
    Column('acc_z', Integer),
    Column('user_id', Integer),
)


def upgrade(migrate_engine):
    # Upgrade operations go here. Don't create your own engine; bind
    # migrate_engine to your metadata
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    post_meta.tables['user'].create()
    pre_meta.tables['reading'].columns['email'].drop()
    pre_meta.tables['reading'].columns['user'].drop()
    post_meta.tables['reading'].columns['alt_timestamp'].create()
    post_meta.tables['reading'].columns['user_id'].create()


def downgrade(migrate_engine):
    # Operations to reverse the above upgrade go here.
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    post_meta.tables['user'].drop()
    pre_meta.tables['reading'].columns['email'].create()
    pre_meta.tables['reading'].columns['user'].create()
    post_meta.tables['reading'].columns['alt_timestamp'].drop()
    post_meta.tables['reading'].columns['user_id'].drop()
