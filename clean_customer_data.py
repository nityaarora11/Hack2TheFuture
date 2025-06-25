import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Download stopwords if not already done
# import nltk
# nltk.download('stopwords')

sf_terms = set([
    'opportunity', 'case', 'soql', 'apex', 'lightning', 'salesforce', 'object', 'field', 'record', 'api', 'error',
    'workflow', 'trigger', 'dashboard', 'report', 'lead', 'contact', 'account', 'custom', 'validation', 'rule',
    'deployment', 'sandbox', 'production', 'org', 'id', 'recordtype', 'profile', 'permission', 'flow', 'queue',
    'chatter', 'community', 'visualforce', 'lwc', 'aura', 'vf', 'sobject', 'lookup', 'master-detail', 'sharing',
    'security', 'role', 'user', 'admin', 'metadata', 'package', 'release', 'update', 'upgrade', 'integration',
    'rest', 'soap', 'bulk', 'event', 'platform', 'limit', 'governor', 'batch', 'schedule', 'cron', 'email',
    'template', 'approval', 'process', 'assignment', 'escalation', 'milestone', 'entitlement', 'sla', 'omni',
    'channel', 'console', 'service', 'cloud', 'sales', 'marketing', 'cpq', 'einstein', 'analytics', 'wave',
    'tableau', 'heroku', 'mulesoft', 'commerce', 'pardot', 'b2b', 'b2c', 'experience', 'site', 'page', 'component',
    'app', 'builder', 'mobile', 'sdk', 'canvas', 'oauth', 'jwt', 'sso', 'login', 'logout', 'session', 'timeout',
    'token', 'refresh', 'callback', 'redirect', 'url', 'endpoint', 'webhook', 'callout', 'external', 'system',
    'data', 'import', 'export', 'loader', 'dataloader', 'datamigration', 'mapping', 'schema', 'relationship',
    'picklist', 'multiselect', 'checkbox', 'currency', 'date', 'datetime', 'time', 'number', 'percent', 'text',
    'textarea', 'richtext', 'encrypted', 'formula', 'rollup', 'summary', 'auto', 'unique', 'required', 'default',
    'value', 'help', 'description', 'label', 'name', 'namespace', 'prefix', 'suffix', 'version', 'patch', 'hotfix',
    'bug', 'issue', 'fix', 'defect', 'incident', 'problem', 'root', 'cause', 'analysis', 'rca', 'workaround',
    'solution', 'resolution', 'status', 'open', 'closed', 'pending', 'inprogress', 'escalated', 'assigned',
    'unassigned', 'owner', 'group', 'team', 'collaboration', 'comment', 'note', 'attachment', 'file', 'document',
    'content', 'library', 'folder', 'access', 'login', 'logout', 'session', 'timeout', 'token', 'refresh',
    'callback', 'redirect', 'url', 'endpoint', 'webhook', 'callout', 'integration', 'external', 'system', 'import',
    'export', 'loader', 'dataloader', 'datamigration', 'mapping', 'schema', 'relationship', 'picklist', 'multiselect',
    'checkbox', 'currency', 'date', 'datetime', 'time', 'number', 'percent', 'text', 'textarea', 'richtext',
    'encrypted', 'formula', 'rollup', 'summary', 'auto', 'unique', 'required', 'default', 'value', 'help',
    'description', 'label', 'name', 'namespace', 'prefix', 'suffix', 'version', 'release', 'patch', 'hotfix', 'bug',
    'issue', 'fix', 'defect', 'incident', 'problem', 'root', 'cause', 'analysis', 'rca', 'workaround', 'solution',
    'resolution', 'status', 'open', 'closed', 'pending', 'inprogress', 'escalated', 'assigned', 'unassigned',
    'owner', 'group', 'team', 'collaboration', 'comment', 'note', 'attachment', 'file', 'document', 'content',
    'library', 'folder', 'access', 'permission', 'profile', 'role', 'user', 'admin'
])

stop_words = set(stopwords.words('english'))
salutations = ['hi', 'hello', 'dear', 'greetings', 'good morning', 'good afternoon', 'good evening', 'to whom it may concern']

def clean_text(text):
    if pd.isna(text):
        return text
    text = str(text).strip().lower()
    # Remove salutations
    for sal in salutations:
        if text.startswith(sal):
            text = text[len(sal):].strip()
    # Remove stop words except Salesforce terms
    words = re.findall(r'\w+', text)
    cleaned_words = [w for w in words if (w in sf_terms or w not in stop_words)]
    return ' '.join(cleaned_words)

df = pd.read_csv('customer_support_data.csv', encoding='latin1')
for col in ['Case Comments', 'Subject', 'Description']:
    if col in df.columns:
        df[col] = df[col].apply(clean_text)
df.to_csv('cleaned_customer_data.csv', index=False) 