from sqlalchemy import *
from migrate import *


from migrate.changeset import schema
pre_meta = MetaData()
post_meta = MetaData()
reading = Table('reading', pre_meta,
    Column('id', INTEGER, primary_key=True, nullable=False),
    Column('device', VARCHAR(length=120)),
    Column('timestamp', VARCHAR(length=120)),
    Column('acc_x', INTEGER),
    Column('acc_y', INTEGER),
    Column('acc_z', INTEGER),
    Column('alt_timestamp', DATETIME),
    Column('user_id', INTEGER),
)

post = Table('post', post_meta,
    Column('id', Integer, primary_key=True, nullable=False),
    Column('body', String(length=140)),
    Column('timestamp', DateTime),
    Column('user_id', Integer),
)


def upgrade(migrate_engine):
    # Upgrade operations go here. Don't create your own engine; bind
    # migrate_engine to your metadata
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    pre_meta.tables['reading'].drop()
    post_meta.tables['post'].create()


def downgrade(migrate_engine):
    # Operations to reverse the above upgrade go here.
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    pre_meta.tables['reading'].create()
    post_meta.tables['post'].drop()
