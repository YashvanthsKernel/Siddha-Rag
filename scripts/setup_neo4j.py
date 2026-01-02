"""
Neo4j Setup and Initialization Script
Sets up the Neo4j database schema for Siddha RAG system.
"""

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from pathlib import Path
import argparse
import sys


def test_connection(uri: str, user: str, password: str) -> bool:
    """
    Test connection to Neo4j database.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        print(f"✅ Successfully connected to Neo4j at {uri}")
        return True
    except ServiceUnavailable:
        print(f"❌ Cannot connect to Neo4j at {uri}")
        print("   Please ensure Neo4j is running")
        return False
    except AuthError:
        print(f"❌ Authentication failed for user: {user}")
        print("   Please check your credentials")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def create_schema(driver):
    """
    Execute the schema.cypher file to create constraints and indexes.
    
    Args:
        driver: Neo4j driver instance
    """
    # Load schema file
    schema_path = Path(__file__).parent.parent / "src" / "graph" / "schema.cypher"
    
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        return False
    
    print(f"📄 Loading schema from: {schema_path}")
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_content = f.read()
    
    # Split into individual statements (separated by semicolons)
    statements = [
        stmt.strip() 
        for stmt in schema_content.split(';') 
        if stmt.strip() and not stmt.strip().startswith('//')
    ]
    
    print(f"🔧 Executing {len(statements)} schema statements...")
    
    success_count = 0
    error_count = 0
    
    with driver.session() as session:
        for i, statement in enumerate(statements, 1):
            # Skip comments and empty statements
            if statement.startswith('//') or statement.startswith('/*') or len(statement) < 10:
                continue
            
            try:
                session.run(statement)
                success_count += 1
                print(f"  ✓ Statement {i}/{len(statements)}")
            except Exception as e:
                error_count += 1
                print(f"  ⚠️ Statement {i} warning: {str(e)[:100]}")
                # Continue even if constraint already exists
    
    print(f"\n✅ Schema setup complete:")
    print(f"   • Successful: {success_count}")
    print(f"   • Warnings: {error_count} (often OK if constraints already exist)")
    
    return True


def verify_setup(driver):
    """
    Verify that schema was created successfully.
    
    Args:
        driver: Neo4j driver instance
    """
    print("\n🔍 Verifying setup...")
    
    # Check constraints
    cypher_constraints = "SHOW CONSTRAINTS"
    
    with driver.session() as session:
        try:
            result = session.run(cypher_constraints)
            constraints = list(result)
            print(f"   • Constraints: {len(constraints)}")
            
            # Check indexes
            cypher_indexes = "SHOW INDEXES"
            result = session.run(cypher_indexes)
            indexes = list(result)
            print(f"   • Indexes: {len(indexes)}")
            
            # Count nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()['count']
            print(f"   • Current nodes: {node_count}")
            
            print("\n✅ Verification complete!")
            
        except Exception as e:
            print(f"⚠️ Verification warning: {e}")


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Set up Neo4j database for Siddha RAG system"
    )
    parser.add_argument(
        '--uri',
        default='bolt://localhost:7687',
        help='Neo4j connection URI (default: bolt://localhost:7687)'
    )
    parser.add_argument(
        '--user',
        default='neo4j',
        help='Neo4j username (default: neo4j)'
    )
    parser.add_argument(
        '--password',
        required=True,
        help='Neo4j password (required)'
    )
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip verification step'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 Siddha RAG - Neo4j Database Setup")
    print("="*70)
    print()
    
    # Step 1: Test connection
    print("Step 1: Testing connection...")
    if not test_connection(args.uri, args.user, args.password):
        print("\n❌ Setup failed: Cannot connect to Neo4j")
        print("\nTroubleshooting:")
        print("  1. Make sure Neo4j is running (check Neo4j Desktop or service)")
        print("  2. Verify the URI is correct (default: bolt://localhost:7687)")
        print("  3. Check your username and password")
        sys.exit(1)
    
    # Step 2: Create schema
    print("\nStep 2: Creating schema...")
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    
    try:
        if not create_schema(driver):
            print("\n⚠️ Schema creation had warnings, but may be OK")
        
        # Step 3: Verify setup
        if not args.skip_verify:
            verify_setup(driver)
        
        print("\n" + "="*70)
        print("✅ Neo4j setup complete!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Verify schema in Neo4j Browser: http://localhost:7474")
        print("  2. Run: python scripts/migrate_to_graph.py")
        print("  3. Start using hybrid retrieval in your RAG system")
        
    finally:
        driver.close()


if __name__ == "__main__":
    main()
