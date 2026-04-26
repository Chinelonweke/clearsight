"""add patient auth - password_hash and password_reset_tokens

Revision ID: a1b2c3d4e5f6
Revises: 580f750e3f43
Create Date: 2026-04-26
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
import uuid

# revision identifiers
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '580f750e3f43'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add password_hash to patients table
    op.add_column(
        'patients',
        sa.Column('password_hash', sa.String(255), nullable=True)
    )

    # Create password_reset_tokens table
    op.create_table(
        'password_reset_tokens',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column(
            'patient_id', UUID(as_uuid=True),
            sa.ForeignKey('patients.id', ondelete='CASCADE'),
            nullable=False, index=True
        ),
        sa.Column('token', sa.String(128), unique=True, nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('used', sa.Boolean(), default=False, nullable=False,
                  server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
    )
    op.create_index('ix_password_reset_tokens_token', 'password_reset_tokens', ['token'], if_not_exists=True)
    op.create_index('ix_password_reset_tokens_patient_id', 'password_reset_tokens', ['patient_id'], if_not_exists=True)


def downgrade() -> None:
    op.drop_index('ix_password_reset_tokens_patient_id', 'password_reset_tokens')
    op.drop_index('ix_password_reset_tokens_token', 'password_reset_tokens')
    op.drop_table('password_reset_tokens')
    op.drop_column('patients', 'password_hash')
